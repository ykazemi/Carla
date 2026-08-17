"""Prisoner's Dilemma payoff structures and the stepwise IPD environment.

Mirrors the paper's §III-B formalism: PD is parameterized by (R, S, T, P)
["reward", "sucker", "temptation", "punishment"]. Two parameterizations are
used across the experiments:

- Donor/recipient (Experiment I, varying dilemma strength r):
    R = 1, T = 1 + r, P = 0, S = -r
- Classic (Experiments II-IV):
    R = 3, S = 0, T = 5, P = 1

Actions are represented as 0 = Cooperate, 1 = Defect throughout this
package (independent of axelrod's own `Action` enum, which is only used at
the boundary in `strategies.py`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

COOPERATE = 0
DEFECT = 1


@dataclass(frozen=True)
class PayoffMatrix:
    """R, S, T, P payoffs for a symmetric 2-action Prisoner's Dilemma."""

    R: float
    S: float
    T: float
    P: float

    def rewards(self, a_i: int, a_j: int) -> tuple[float, float]:
        """Return (reward_i, reward_j) for the joint action (a_i, a_j)."""
        if a_i == COOPERATE and a_j == COOPERATE:
            return self.R, self.R
        if a_i == COOPERATE and a_j == DEFECT:
            return self.S, self.T
        if a_i == DEFECT and a_j == COOPERATE:
            return self.T, self.S
        return self.P, self.P

    @classmethod
    def donor_recipient(cls, r: float) -> "PayoffMatrix":
        """R=1, T=1+r, P=0, S=-r, per eq. (1) and §V-B."""
        return cls(R=1.0, S=-r, T=1.0 + r, P=0.0)

    @classmethod
    def classic(cls) -> "PayoffMatrix":
        """R=3, S=0, T=5, P=1, used in Experiments II-IV."""
        return cls(R=3.0, S=0.0, T=5.0, P=1.0)


def apply_noise(action: int, noise_level: float, rng: np.random.Generator) -> int:
    """Flip an action with probability `noise_level` (§IV-A / §V-D)."""
    if noise_level > 0 and rng.random() < noise_level:
        return DEFECT if action == COOPERATE else COOPERATE
    return action


class IPDEnv:
    """Drives one iterated PD match turn by turn.

    This class only computes rewards from a joint action pair; it does not
    own either player's decision policy. Noise (if any) is applied to both
    actions *before* rewards are computed and before either side's history
    is updated, matching "noise ... applied to all plays after they are
    delivered by the player" (§V-D).
    """

    def __init__(self, payoff: PayoffMatrix, noise_level: float = 0.0, seed: int | None = None):
        self.payoff = payoff
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)

    def step(self, a_i: int, a_j: int) -> tuple[int, int, float, float]:
        """Apply noise, compute rewards.

        Returns (a_i_actual, a_j_actual, reward_i, reward_j) — the
        post-noise actions are what both sides should record into their
        histories, since noise is applied "after [plays] are delivered".
        """
        a_i_actual = apply_noise(a_i, self.noise_level, self.rng)
        a_j_actual = apply_noise(a_j, self.noise_level, self.rng)
        r_i, r_j = self.payoff.rewards(a_i_actual, a_j_actual)
        return a_i_actual, a_j_actual, r_i, r_j


class MetaRandomOpponent:
    """The Experiment IV meta-strategy opponent (§V-E / Table I).

    Each round, every pooled strategy is polled for its prescribed next
    move given the REAL joint history so far; one proposal is picked
    uniformly at random as this round's actual action (that strategy's name
    is the round's ground-truth label). The actual action is then synced
    into every pooled strategy's history so future proposals stay grounded
    in what really happened, not in each strategy's own hypothetical path.
    """

    def __init__(self, pool: dict[str, "SteppablePlayer"], seed: int | None = None):
        self.pool = pool
        self.rng = np.random.default_rng(seed)

    def act(self) -> tuple[int, str]:
        proposals = {name: player.act() for name, player in self.pool.items()}
        source = self.rng.choice(list(proposals.keys()))
        return proposals[source], source

    def observe(self, my_action: int, carla_action: int) -> None:
        for player in self.pool.values():
            player.observe(my_action, carla_action)

    def reset(self) -> None:
        for player in self.pool.values():
            player.reset()
