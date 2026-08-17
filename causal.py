"""Causal influence reward: counterfactual marginalization + KL (eq. 12-13).

Two documented deviations from a literal reading of the paper (see README):

- Eq. (12)'s counterfactual-averaging weight is implemented UNCONDITIONED on
  a_j^t (`sum_a pi_theta(a|s) * P(a_j^{t+1}|a,s,phi)`), matching the prose
  ("if we sample many counterfactual actions [a~ ~ P(a_i^t|s^t)]") and
  Fig. 2's actual network input (theta_ij never takes a_j^t as input). The
  equation as OCR'd conditions on a_j^t, which is inconsistent with both.
- Since |A|=2, the "sample many counterfactual actions" marginalization is
  computed by exact enumeration over {Cooperate, Defect} rather than Monte
  Carlo sampling -- equivalent in expectation, cheaper, and exact.
"""
from __future__ import annotations

import torch

from carla.networks import OpponentModelNet, onehot_action

EPS = 1e-3
KL_CLIP = 5.0


def counterfactual_marginal(policy_probs: torch.Tensor, window: torch.Tensor,
                             opponent_model: OpponentModelNet) -> torch.Tensor:
    """Eq. (12): marginal P(a_j^{t+1}|s^t,theta_ij), averaging the opponent
    model's prediction over CARLA's own counterfactual actions ~ pi_theta.

    policy_probs: (B, 2) -- pi_theta_ij(.|s^t)
    window: (B, T, 4) -- the same state window used for both networks
    Returns: (B, 2) marginal distribution over the opponent's next action.
    """
    batch = window.shape[0]
    device = window.device
    p_c = opponent_model(window, onehot_action(torch.zeros(batch, dtype=torch.long, device=device)))
    p_d = opponent_model(window, onehot_action(torch.ones(batch, dtype=torch.long, device=device)))
    return policy_probs[:, 0:1] * p_c + policy_probs[:, 1:2] * p_d


def causal_reward(window: torch.Tensor, a_i_onehot: torch.Tensor, policy_probs: torch.Tensor,
                   opponent_model: OpponentModelNet, eps: float = EPS, clip: float = KL_CLIP) -> torch.Tensor:
    """Eq. (13): c^t_ij = KL[ P(a_j^{t+1}|a_i^t,s^t,phi) || P(a_j^{t+1}|s^t,theta_ij) ].

    High KL means CARLA's actual action a_i^t is much more informative
    about the opponent's predicted next move than CARLA's average
    (counterfactual-marginalized) behavior would be -- i.e. CARLA is
    exerting causal influence over the opponent's prediction.

    Returns: (B,) causal reward per sample, clamped to [0, clip] for
    training stability (KL is unbounded as probabilities approach 0/1).
    """
    p_cond = opponent_model(window, a_i_onehot).clamp(eps, 1 - eps)
    p_marg = counterfactual_marginal(policy_probs, window, opponent_model).clamp(eps, 1 - eps)
    kl = (p_cond * (p_cond.log() - p_marg.log())).sum(dim=-1)
    return kl.clamp(min=0.0, max=clip)


def causal_influence(window: torch.Tensor, policy_probs: torch.Tensor,
                      opponent_model: OpponentModelNet, eps: float = EPS, clip: float = KL_CLIP) -> torch.Tensor:
    """Expectation form of the causal reward: E_{a~pi_theta}[ KL(P(a_j^{t+1}|a,s,phi) || marginal) ].

    This is the mutual-information-style quantity behind Jaques et al.'s
    "social influence as intrinsic motivation" (ref. [28]), which the paper
    cites as its direct inspiration for the causal-reward mechanism ("We
    used a very similar approach regarding the implementation of causal
    reward"). `causal_reward()` above conditions on one specific action
    a_i^t (correct for the per-step TRAINING reward in eq. 13, where a_i^t
    is whatever action was actually sampled that step); this function
    instead averages the KL over both actions weighted by pi_theta,
    matching the Jaques et al. form.

    This distinction matters at INFERENCE time (Algorithm 4): once a
    policy has converged close to deterministic (as these do -- verified
    empirically, see README), conditioning on the single argmax action
    makes p_cond collapse onto p_marg almost exactly, driving the
    causal-reward term to ~0 regardless of how informative the opponent
    model actually is -- silently defeating the ensemble's opponent-
    detection mechanism. Weighting by pi_theta instead keeps both actions'
    KL contributions visible even when one has very low probability.
    """
    batch = window.shape[0]
    device = window.device
    p_c = opponent_model(window, onehot_action(torch.zeros(batch, dtype=torch.long, device=device))).clamp(eps, 1 - eps)
    p_d = opponent_model(window, onehot_action(torch.ones(batch, dtype=torch.long, device=device))).clamp(eps, 1 - eps)
    p_marg = (policy_probs[:, 0:1] * p_c + policy_probs[:, 1:2] * p_d).clamp(eps, 1 - eps)
    kl_c = (p_c * (p_c.log() - p_marg.log())).sum(dim=-1)
    kl_d = (p_d * (p_d.log() - p_marg.log())).sum(dim=-1)
    expected_kl = policy_probs[:, 0] * kl_c + policy_probs[:, 1] * kl_d
    return expected_kl.clamp(min=0.0, max=clip)


def opponent_model_fit(window_before: torch.Tensor, a_i_last_onehot: torch.Tensor,
                        a_j_last: torch.Tensor, opponent_model: OpponentModelNet, eps: float = EPS) -> torch.Tensor:
    """Log-likelihood of member j's opponent model phi_j correctly
    predicting the REALLY observed opponent action a_j_last, given CARLA's
    really observed previous action a_i_last and the window before that
    turn. Used as the ensemble's opponent-IDENTIFICATION signal at
    inference time (see README) -- unlike causal_reward/causal_influence
    above, this doesn't depend at all on how (near-)deterministic
    theta_ij's policy has become, since it only evaluates phi_j (trained by
    ordinary supervised CE, see README deviation 1) against a real,
    already-realized transition. Empirically this is a far cleaner
    detection signal than the causal-influence terms once policies have
    converged close to deterministic.
    """
    pred = opponent_model(window_before, a_i_last_onehot).clamp(eps, 1 - eps)
    return torch.log(pred.gather(1, a_j_last.unsqueeze(1)).squeeze(1))
