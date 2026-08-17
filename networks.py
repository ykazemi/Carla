"""CARLA's per-strategy networks (Fig. 2 / Fig. 3).

Each Axelrod strategy j gets its own pair of networks:

- `ActorCriticNet` (theta_ij): FC0 -> FC1 -> LSTM -> {policy head, value
  head}. Trained via advantage actor-critic (§III-C, eq. 8-10).
- `OpponentModelNet` (phi_j): same FC0 -> FC1 -> LSTM backbone shape, but a
  *separate* set of weights (Fig. 3: "Both models share the same input, but
  each has its own set of fully connected layers"), plus CARLA's own action
  a_i^t concatenated in at the head, predicting P(a_j^{t+1} | a_i^t, s^t).

State representation: a fixed-length sliding window of the last `n` turns'
joint actions, one-hot encoded per turn as [C_i, D_i, C_j, D_j] (4 dims per
timestep), fed through the LSTM. This truncated-window design is stated
explicitly for phi in §IV-C ("a list of action pairs up to n previous
interactions") and applied uniformly to theta_ij here for implementation
consistency (see README reproduction notes).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_DIM = 2  # Cooperate=0, Defect=1
TURN_DIM = 2 * ACTION_DIM  # one-hot(a_i) concat one-hot(a_j) per turn


def onehot_action(actions: torch.Tensor, num_classes: int = ACTION_DIM) -> torch.Tensor:
    """actions: (...,) long tensor of 0/1 -> (..., num_classes) float tensor."""
    return F.one_hot(actions.long(), num_classes=num_classes).float()


class RecurrentBackbone(nn.Module):
    """FC0 -> ReLU -> FC1 -> ReLU -> LSTM, returning the final hidden state.

    hidden_size=14 follows the paper's "14 LSTM layers" (§IV-B), read as an
    LSTM hidden width of 14 with a single layer -- 14 literally-stacked
    LSTM layers is infeasible/unmotivated for a 2-action game and matches
    nothing else numeric in the paper (see README reproduction notes).
    """

    def __init__(self, input_dim: int = TURN_DIM, fc0: int = 64, fc1: int = 32, lstm_hidden: int = 14):
        super().__init__()
        self.fc0 = nn.Linear(input_dim, fc0)
        self.fc1 = nn.Linear(fc0, fc1)
        self.lstm = nn.LSTM(input_size=fc1, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
        self.hidden_size = lstm_hidden

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """window: (B, T, input_dim) -> (B, hidden_size), the LSTM's final hidden state."""
        x = F.relu(self.fc0(window))
        x = F.relu(self.fc1(x))
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]  # (B, hidden_size)


class ActorCriticNet(nn.Module):
    """Policy (actor) + value (critic) heads over a shared recurrent backbone."""

    def __init__(self, **backbone_kwargs):
        super().__init__()
        self.backbone = RecurrentBackbone(**backbone_kwargs)
        self.policy_head = nn.Linear(self.backbone.hidden_size, ACTION_DIM)
        self.value_head = nn.Linear(self.backbone.hidden_size, 1)

    def forward(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy_probs (B, 2), value (B,))."""
        h = self.backbone(window)
        probs = F.softmax(self.policy_head(h), dim=-1)
        value = self.value_head(h).squeeze(-1)
        return probs, value


class OpponentModelNet(nn.Module):
    """Predicts P(a_j^{t+1} | a_i^t, s^t, phi) -- the "model of other player's action" (Fig. 3, bottom).

    A separate backbone from ActorCriticNet (own FC0/FC1/LSTM weights);
    CARLA's own action a_i^t is concatenated at the head, since it is not
    part of the history window s^t = H^{t-1} (which only covers turns
    strictly before t).
    """

    def __init__(self, **backbone_kwargs):
        super().__init__()
        self.backbone = RecurrentBackbone(**backbone_kwargs)
        self.head = nn.Linear(self.backbone.hidden_size + ACTION_DIM, ACTION_DIM)

    def forward(self, window: torch.Tensor, a_i_onehot: torch.Tensor) -> torch.Tensor:
        """window: (B, T, input_dim); a_i_onehot: (B, 2) -> P(a_j^{t+1}|.) (B, 2)."""
        h = self.backbone(window)
        logits = self.head(torch.cat([h, a_i_onehot], dim=-1))
        return F.softmax(logits, dim=-1)
