"""Builds the full (N+1)x(N+1) round-robin payoff matrix (Figs. 4-6): a fast
`axelrod.Tournament` among the fixed strategies themselves, combined with
CARLA's RL-driven matches (`carla.train.evaluate_repeated`) for the row/
column CARLA participates in.

Scoring convention (documented assumption, see README): a player's "mean
score" is the row-mean of the payoff matrix EXCLUDING the diagonal
(self-play) -- this is well-defined uniformly for every player including
CARLA, which has no self-play entry.
"""
from __future__ import annotations

import axelrod as axl
import numpy as np
import pandas as pd

from carla.agent import CARLAAgent
from carla.config import Config
from carla.game import PayoffMatrix
from carla.strategies import STRATEGY_REGISTRY
from carla.train import evaluate_repeated

CARLA_LABEL = "CARLA"


def axl_game(payoff: PayoffMatrix) -> axl.Game:
    return axl.Game(r=payoff.R, s=payoff.S, t=payoff.T, p=payoff.P)


def fixed_strategy_tournament(names: list[str], payoff: PayoffMatrix, noise_level: float,
                               turns: int, repetitions: int, seed: int):
    """Round-robin among the fixed (non-CARLA) strategies. Returns the axelrod ResultSet."""
    players = [STRATEGY_REGISTRY[name]() for name in names]
    game = axl_game(payoff)
    tournament = axl.Tournament(players, game=game, turns=turns, repetitions=repetitions,
                                 noise=noise_level, seed=seed)
    return tournament.play(progress_bar=False)


def full_payoff_matrix(agent: CARLAAgent, names: list[str], payoff: PayoffMatrix, noise_level: float,
                        config: Config, seed: int) -> pd.DataFrame:
    """(N+1)x(N+1) matrix of mean payoffs, row-player's score vs col-player, CARLA included."""
    results = fixed_strategy_tournament(names, payoff, noise_level, config.eval_turns, config.eval_repeats, seed)
    matrix = np.array(results.payoff_matrix, dtype=float)
    full = pd.DataFrame(matrix, index=names, columns=names)
    full[CARLA_LABEL] = np.nan
    full.loc[CARLA_LABEL] = np.nan

    for name in names:
        matches = evaluate_repeated(agent, name, payoff, noise_level, config, base_seed=seed + 5000)
        full.loc[CARLA_LABEL, name] = float(np.mean([m["mean_score_carla"] for m in matches]))
        full.loc[name, CARLA_LABEL] = float(np.mean([m["mean_score_opponent"] for m in matches]))
    return full


def mean_scores(matrix: pd.DataFrame) -> pd.Series:
    """Each player's mean score against all OTHER players (diagonal excluded)."""
    values = matrix.astype(float).copy()
    np.fill_diagonal(values.values, np.nan)
    return values.mean(axis=1, skipna=True)


def score_distributions(agent: CARLAAgent, names: list[str], payoff: PayoffMatrix, noise_level: float,
                         config: Config, seed: int) -> pd.DataFrame:
    """Per-repetition mean-score samples for the Fig. 5/7 box plots.

    Fixed strategies use axelrod's own `normalised_scores` (their mean
    score per repetition across the OTHER FIXED strategies only -- this
    does not include their single match against CARLA; documented as a
    minor simplification in README). CARLA's samples are its mean score
    per repetition across all fixed-strategy opponents (matching repeat
    index r across opponents via consistent seeding).
    """
    results = fixed_strategy_tournament(names, payoff, noise_level, config.eval_turns, config.eval_repeats, seed)
    # `results.players`/`normalised_scores` preserve the order `players` was
    # constructed in (== `names`), so zip against `names` directly rather
    # than axelrod's own str(player) repr (which includes parameter
    # suffixes, e.g. "ZD-Extortion: 0.2, 0.1, 1", not our canonical labels).
    rows = []
    for name, scores in zip(names, results.normalised_scores):
        for s in scores:
            rows.append({"strategy": name, "score": s})

    per_opponent = {
        name: evaluate_repeated(agent, name, payoff, noise_level, config, base_seed=seed + 5000)
        for name in names
    }
    for r in range(config.eval_repeats):
        carla_r_score = float(np.mean([per_opponent[name][r]["mean_score_carla"] for name in names]))
        rows.append({"strategy": CARLA_LABEL, "score": carla_r_score})

    return pd.DataFrame(rows)
