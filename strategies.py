"""Axelrod tournament strategy registry and a turn-by-turn adapter.

`axelrod.Match` normally owns an entire match's turn loop internally. CARLA
needs to interleave its own online learning with each turn, so
`SteppablePlayer` exposes the same underlying mechanics
(`Player.strategy(opponent)` + `Player.update_history(play, coplay)`) one
turn at a time. This mirrors exactly what `Match.simultaneous_play` does
internally (verified against `axelrod.Match` directly for several
strategies before relying on it — see tests/test_strategies.py).

Action convention: this module speaks in axelrod.Action (C/D) at its
boundary and converts to/from the package-wide int convention
(0=Cooperate, 1=Defect) from `carla.game`.
"""
from __future__ import annotations

from typing import Callable

import axelrod as axl

from carla.game import COOPERATE, DEFECT

# Maps our canonical strategy names (matching the paper's Figs. 4-8 axis
# labels) to a zero-arg factory constructing a fresh axelrod.Player.
# Grumpy(Nice, 10, -10) and ZDExtortion(0.2, 0.1, 1) use axelrod's own
# defaults, which were verified interactively to already match the paper's
# stated parameters exactly.
STRATEGY_REGISTRY: dict[str, Callable[[], axl.Player]] = {
    "Cooperator": axl.Cooperator,
    "Defector": axl.Defector,
    "Tit For Tat": axl.TitForTat,
    "Alternator": axl.Alternator,
    "Bully": axl.Bully,
    "Anti Tit For Tat": axl.AntiTitForTat,
    "Cycler DC": axl.CyclerDC,
    "Suspicious Tit For Tat": axl.SuspiciousTitForTat,
    "Win-Stay Lose-Shift": axl.WinStayLoseShift,
    "Grumpy": axl.Grumpy,
    "ZD-Extortion": axl.ZDExtortion,
}

# The 11 handcrafted strategies, in the order used throughout Figs. 4-8.
HANDCRAFTED_STRATEGIES: list[str] = list(STRATEGY_REGISTRY.keys())

# Experiments II-IV additionally include the pretrained EvolvedANN5 (§II,
# ref. [30]); kept in a separate registry entry since it's excluded from
# Experiment I (whose axis in Fig. 4 lists only the 11 handcrafted ones).
EVOLVED_ANN_NAME = "EvolvedANN5"
STRATEGY_REGISTRY[EVOLVED_ANN_NAME] = axl.EvolvedANN5

ALL_STRATEGIES: list[str] = HANDCRAFTED_STRATEGIES + [EVOLVED_ANN_NAME]


def _to_axl(action: int) -> axl.Action:
    return axl.Action.C if action == COOPERATE else axl.Action.D


def _from_axl(action: axl.Action) -> int:
    return COOPERATE if action == axl.Action.C else DEFECT


class SteppablePlayer:
    """Turn-by-turn wrapper around an axelrod.Player.

    `act()` asks the wrapped strategy for its next move given the real
    joint history observed so far (via a bare `axl.Player()` proxy standing
    in for CARLA, whose `.history` is kept in sync by `observe()`).
    `observe()` must be called exactly once per turn, after both actions
    for that turn are known, to advance both histories.
    """

    def __init__(self, name: str, seed: int | None = None):
        if name not in STRATEGY_REGISTRY:
            raise KeyError(f"Unknown strategy '{name}'. Known: {list(STRATEGY_REGISTRY)}")
        self.name = name
        self._factory = STRATEGY_REGISTRY[name]
        self._seed = seed
        self.player: axl.Player = self._factory()
        self.opponent_proxy: axl.Player = axl.Player()
        self.reset()

    def reset(self) -> None:
        self.player = self._factory()
        self.opponent_proxy = axl.Player()
        self.player.reset()
        if axl.Classifiers["stochastic"](self.player) and self._seed is not None:
            self.player.set_seed(self._seed)

    def act(self) -> int:
        """Return this strategy's next action (int) given history so far."""
        return _from_axl(self.player.strategy(self.opponent_proxy))

    def observe(self, my_action: int, carla_action: int) -> None:
        """Record the turn's actual (this-strategy, CARLA) action pair."""
        self.player.update_history(_to_axl(my_action), _to_axl(carla_action))
        self.opponent_proxy.update_history(_to_axl(carla_action), _to_axl(my_action))


def make_pool(names: list[str], seed: int | None = None) -> dict[str, SteppablePlayer]:
    """Build a fresh {name: SteppablePlayer} pool, e.g. for MetaRandomOpponent."""
    return {name: SteppablePlayer(name, seed=seed) for name in names}
