"""CARLA: the ensemble causal-RL agent (Algorithms 1-4).

Per opponent strategy j, CARLA owns an independent (actor-critic theta_ij,
opponent-model phi_j) pair (`StrategySlot`). `train_strategy` implements
the interleaved Algorithm 1 + Algorithm 2 training loop; `ensemble_act`
implements Algorithm 3 + 4 inference-time action selection.

See README.md "Reproduction Notes" for the documented deviations from a
literal reading of the paper's algorithm boxes (opponent-model training
objective, actor-critic target formula, counterfactual-marginalization
conditioning, and the V_j substitution used in Algorithm 4's scoring rule).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from carla.causal import causal_influence, causal_reward, opponent_model_fit
from carla.config import Config
from carla.game import IPDEnv, PayoffMatrix
from carla.networks import ActorCriticNet, OpponentModelNet, onehot_action
from carla.replay import OpponentReplayBuffer
from carla.strategies import SteppablePlayer

ZERO_TURN = (0.0, 0.0, 0.0, 0.0)


def turn_vector(a_i: int, a_j: int) -> tuple[float, float, float, float]:
    v = [0.0, 0.0, 0.0, 0.0]
    v[a_i] = 1.0
    v[2 + a_j] = 1.0
    return tuple(v)


def empty_window(window_length: int) -> deque:
    return deque([ZERO_TURN] * window_length, maxlen=window_length)


def window_tensor(history: deque) -> torch.Tensor:
    return torch.tensor(list(history), dtype=torch.float32).unsqueeze(0)  # (1, T, 4)


@dataclass
class Trajectory:
    windows: torch.Tensor       # (T, window_length, 4)
    next_windows: torch.Tensor  # (T, window_length, 4)
    a_i: torch.Tensor           # (T,) long
    a_j: torch.Tensor           # (T,) long
    e: torch.Tensor             # (T,) float, environmental reward e^t
    total_env_reward: float


@dataclass
class StrategySlot:
    actor_critic: ActorCriticNet
    opponent_model: OpponentModelNet
    ac_optim: torch.optim.Optimizer
    om_optim: torch.optim.Optimizer
    replay: OpponentReplayBuffer
    value_baseline: float = 0.0


class CARLAAgent:
    def __init__(self, strategy_names: list[str], config: Config):
        self.strategy_names = list(strategy_names)
        self.config = config
        self.slots: dict[str, StrategySlot] = {}
        for name in self.strategy_names:
            ac = ActorCriticNet()
            om = OpponentModelNet()
            self.slots[name] = StrategySlot(
                actor_critic=ac,
                opponent_model=om,
                ac_optim=torch.optim.Adam(ac.parameters(), lr=config.learning_rate),
                om_optim=torch.optim.Adam(om.parameters(), lr=config.learning_rate),
                replay=OpponentReplayBuffer(config.opponent_replay_capacity),
            )

    # ------------------------------------------------------------------
    # Algorithm 1 (rollout half) shared by training and evaluation.
    # ------------------------------------------------------------------
    def _rollout(self, opponent: SteppablePlayer, actor_critic: ActorCriticNet,
                 env: IPDEnv, n_turns: int, window_length: int,
                 sample: bool, torch_rng: torch.Generator | None) -> Trajectory:
        opponent.reset()
        history = empty_window(window_length)
        windows, next_windows, a_i_list, a_j_list, e_list = [], [], [], [], []
        for _ in range(n_turns):
            window_t = window_tensor(history)
            with torch.no_grad():
                probs, _ = actor_critic(window_t)
            if sample:
                a_i = int(torch.multinomial(probs[0], 1, generator=torch_rng).item())
            else:
                a_i = int(probs[0].argmax().item())
            a_j = opponent.act()
            a_i_actual, a_j_actual, r_i, _r_j = env.step(a_i, a_j)
            opponent.observe(a_j_actual, a_i_actual)

            windows.append(window_t.squeeze(0))
            history.append(turn_vector(a_i_actual, a_j_actual))
            next_windows.append(window_tensor(history).squeeze(0))
            a_i_list.append(a_i_actual)
            a_j_list.append(a_j_actual)
            e_list.append(r_i)

        return Trajectory(
            windows=torch.stack(windows),
            next_windows=torch.stack(next_windows),
            a_i=torch.tensor(a_i_list, dtype=torch.long),
            a_j=torch.tensor(a_j_list, dtype=torch.long),
            e=torch.tensor(e_list, dtype=torch.float32),
            total_env_reward=float(sum(e_list)),
        )

    # ------------------------------------------------------------------
    # Algorithm 2 (opponent-model half): supervised CE against the
    # actually observed a_j^{t+1}, NOT literal minimization of eq. (13)
    # (see README "Reproduction Notes" -- minimizing the causal-influence
    # KL directly would train phi to be uninformative about CARLA's
    # action, destroying the signal the rest of the system depends on).
    # ------------------------------------------------------------------
    def _update_opponent_model(self, slot: StrategySlot, traj: Trajectory, config: Config) -> float | None:
        if traj.windows.shape[0] < 2:
            return None
        windows = traj.windows[:-1]
        a_i = traj.a_i[:-1]
        labels = traj.a_j[1:]  # a_j^{t+1}
        slot.replay.push_batch(windows, a_i, labels)
        if len(slot.replay) == 0:
            return None
        w, a, y = slot.replay.sample(config.opponent_minibatch)
        pred = slot.opponent_model(w, onehot_action(a))
        loss = F.nll_loss(torch.log(pred.clamp(1e-6, 1.0)), y)
        slot.om_optim.zero_grad()
        loss.backward()
        slot.om_optim.step()
        return float(loss.item())

    # ------------------------------------------------------------------
    # Algorithm 1 (update half): combined reward r^t_ij = alpha*e^t +
    # beta*c^t_ij after `warmup_iters` (eq. 9-10-11), env-only reward
    # during warmup so phi has something to learn from before being
    # trusted (see README).
    # ------------------------------------------------------------------
    def _update_actor_critic(self, slot: StrategySlot, traj: Trajectory, config: Config, use_causal: bool) -> dict:
        with torch.no_grad():
            probs_ng, _ = slot.actor_critic(traj.windows)
            if use_causal:
                c = causal_reward(traj.windows, onehot_action(traj.a_i), probs_ng, slot.opponent_model)
            else:
                c = torch.zeros_like(traj.e)
            r = (config.alpha * traj.e + config.beta * c).detach()
            _, V_next = slot.actor_critic(traj.next_windows)

        probs, V = slot.actor_critic(traj.windows)
        target = (r + config.gamma * V_next).detach()
        advantage = target - V

        log_probs = torch.log(probs.clamp(1e-6, 1.0))
        chosen_log_probs = log_probs.gather(1, traj.a_i.unsqueeze(1)).squeeze(1)
        actor_loss = -(chosen_log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        loss = actor_loss + config.critic_loss_coef * critic_loss - config.entropy_coef * entropy

        slot.ac_optim.zero_grad()
        loss.backward()
        slot.ac_optim.step()
        return {
            "loss": float(loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "mean_causal_reward": float(c.mean().item()),
            "mean_combined_reward": float(r.mean().item()),
        }

    def train_strategy(self, name: str, payoff: PayoffMatrix, noise_level: float,
                        config: Config, log_fn: Callable[[dict], None] | None = None) -> list[dict]:
        slot = self.slots[name]
        env = IPDEnv(payoff, noise_level=noise_level, seed=config.seed)
        torch_rng = torch.Generator().manual_seed(config.seed)
        log = []
        for it in range(config.total_iterations):
            opponent = SteppablePlayer(name, seed=config.seed + it)
            traj = self._rollout(opponent, slot.actor_critic, env, config.train_batch_size,
                                  config.window_length, sample=True, torch_rng=torch_rng)
            om_loss = self._update_opponent_model(slot, traj, config)
            use_causal = it >= config.warmup_iters
            ac_stats = self._update_actor_critic(slot, traj, config, use_causal)
            record = {"iteration": it, "opponent_model_loss": om_loss,
                      "total_env_reward": traj.total_env_reward, **ac_stats}
            log.append(record)
            if log_fn is not None:
                log_fn(record)

        # Calibrate this member's value baseline: V_j(s) is trained to
        # approximate expected return AGAINST OPPONENT j SPECIFICALLY, so
        # its absolute level mostly reflects how exploitable opponent j is
        # (e.g. ~T every turn against Cooperator vs ~P every turn against
        # Defector), not how well s matches opponent j's identity. Without
        # subtracting this per-member level, ensemble_act's argmax always
        # picks whichever member has the highest achievable ceiling,
        # regardless of the actual observed history (verified empirically
        # -- see README). Subtracting each member's own mean V over its
        # final training rollout removes this confound while preserving
        # genuine state-dependent deviation.
        with torch.no_grad():
            _, V_final = slot.actor_critic(traj.windows)
        slot.value_baseline = float(V_final.mean().item())
        return log

    # ------------------------------------------------------------------
    # Algorithm 3 + 4: ensemble inference against a real (unknown)
    # opponent.
    #
    # The primary selection signal is each member's opponent-model FIT to
    # the really-observed last transition (opponent_model_fit): how well
    # phi_j predicts what the opponent actually just did. This is used
    # instead of the causal-influence terms from causal.py for member
    # SELECTION, because of an empirically verified failure mode (see
    # README): once theta_ij's policy is close to deterministic (which
    # happens quickly in this 2-action game even with an entropy bonus),
    # BOTH causal_reward (conditioned on one action) and causal_influence
    # (the expectation form) shrink toward ~0 for every member roughly
    # alike, leaving the ensemble with no usable state-dependent signal --
    # at that point selection was dominated by V_j(s), which mostly
    # reflects each opponent's exploitability ceiling rather than which
    # member fits the observed history. opponent_model_fit has neither
    # problem: it only evaluates the (supervised-trained, see README
    # deviation 1) opponent model against a real transition, so it stays
    # sharply state-dependent regardless of policy convergence. Each
    # member's baseline-subtracted value estimate is kept as a secondary,
    # small-weight term for tie-breaking among members with similar fit.
    # ------------------------------------------------------------------
    def ensemble_act(self, history: deque, rng: np.random.Generator) -> tuple[int, str, dict[str, float]]:
        window = window_tensor(history)
        history_list = list(history)
        has_real_history = history_list[-1] != ZERO_TURN
        if has_real_history:
            # turn_vector(a_i, a_j) sets index a_i in [0,1] and index 2+a_j in [2,3];
            # a_i_last/a_j_last recover the 0=Cooperate/1=Defect indices from that encoding.
            a_i_last = 0 if history_list[-1][0] == 1.0 else 1
            a_j_last = 0 if history_list[-1][2] == 1.0 else 1
            window_before = torch.tensor([[ZERO_TURN] + history_list[:-1]], dtype=torch.float32)
            a_i_last_onehot = onehot_action(torch.tensor([a_i_last]))
            a_j_last_t = torch.tensor([a_j_last])

        scores: dict[str, float] = {}
        proposals: dict[str, int] = {}
        for name, slot in self.slots.items():
            with torch.no_grad():
                probs, V = slot.actor_critic(window)
                a = int(probs[0].argmax().item())
                centered_v = V.item() - slot.value_baseline
                if has_real_history:
                    fit = opponent_model_fit(window_before, a_i_last_onehot, a_j_last_t, slot.opponent_model)
                    score = float(fit.item()) + 0.01 * centered_v
                else:
                    score = centered_v
            scores[name] = score
            proposals[name] = a
        best = max(scores, key=lambda k: (scores[k], rng.random()))
        return proposals[best], best, scores

    def save(self, path) -> None:
        state = {name: {"ac": slot.actor_critic.state_dict(), "om": slot.opponent_model.state_dict(),
                         "value_baseline": slot.value_baseline}
                 for name, slot in self.slots.items()}
        torch.save(state, path)

    def load(self, path) -> None:
        state = torch.load(path, map_location="cpu")
        for name, slot in self.slots.items():
            slot.actor_critic.load_state_dict(state[name]["ac"])
            slot.opponent_model.load_state_dict(state[name]["om"])
            slot.value_baseline = state[name].get("value_baseline", 0.0)
