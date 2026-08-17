"""Training/evaluation orchestration shared by the experiment scripts."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from carla.agent import CARLAAgent, empty_window, turn_vector
from carla.config import Config, load_config
from carla.game import IPDEnv, MetaRandomOpponent, PayoffMatrix
from carla.strategies import SteppablePlayer, make_pool

__all__ = ["load_config", "train_all", "evaluate_match", "evaluate_repeated", "evaluate_meta_strategy"]


def train_all(agent: CARLAAgent, strategy_names: list[str], payoff: PayoffMatrix,
              noise_level: float, config: Config, verbose: bool = True) -> dict[str, list[dict]]:
    """Train every ensemble member (one per opponent strategy), sequentially."""
    logs = {}
    for name in strategy_names:
        t0 = time.time()
        logs[name] = agent.train_strategy(name, payoff, noise_level, config)
        if verbose:
            last = logs[name][-1]
            print(f"  [{name}] {config.total_iterations} iters in {time.time() - t0:.1f}s "
                  f"| final combined reward/turn={last['mean_combined_reward']:.3f}")
    return logs


def evaluate_match(agent: CARLAAgent, opponent_name: str, payoff: PayoffMatrix,
                    noise_level: float, config: Config, seed: int) -> dict:
    """One evaluation match: CARLA (via ensemble_act) vs a fixed, unknown-to-CARLA opponent."""
    opponent = SteppablePlayer(opponent_name, seed=seed)
    env = IPDEnv(payoff, noise_level=noise_level, seed=seed)
    history = empty_window(config.window_length)
    rng = np.random.default_rng(seed)

    total_i, total_j = 0.0, 0.0
    chosen_counts: dict[str, int] = {name: 0 for name in agent.strategy_names}
    for _ in range(config.eval_turns):
        a_i, chosen, _scores = agent.ensemble_act(history, rng)
        a_j = opponent.act()
        a_i_actual, a_j_actual, r_i, r_j = env.step(a_i, a_j)
        opponent.observe(a_j_actual, a_i_actual)
        history.append(turn_vector(a_i_actual, a_j_actual))
        total_i += r_i
        total_j += r_j
        chosen_counts[chosen] += 1

    turns = config.eval_turns
    return {
        "opponent": opponent_name,
        "mean_score_carla": total_i / turns,
        "mean_score_opponent": total_j / turns,
        "chosen_fractions": {k: v / turns for k, v in chosen_counts.items()},
    }


def evaluate_repeated(agent: CARLAAgent, opponent_name: str, payoff: PayoffMatrix,
                       noise_level: float, config: Config, base_seed: int = 1000) -> list[dict]:
    return [
        evaluate_match(agent, opponent_name, payoff, noise_level, config, seed=base_seed + r)
        for r in range(config.eval_repeats)
    ]


def evaluate_meta_strategy(agent: CARLAAgent, pool_names: list[str], payoff: PayoffMatrix,
                            noise_level: float, config: Config, seed: int) -> dict:
    """Experiment IV: CARLA vs the meta-strategy opponent (§V-E).

    Returns per-round (source_label, carla_chosen_label) pairs for the
    confusion matrix, plus mean scores.
    """
    pool = make_pool(pool_names, seed=seed)
    meta = MetaRandomOpponent(pool, seed=seed)
    env = IPDEnv(payoff, noise_level=noise_level, seed=seed)
    history = empty_window(config.window_length)
    rng = np.random.default_rng(seed)

    pairs: list[tuple[str, str]] = []
    total_i, total_j = 0.0, 0.0
    for _ in range(config.eval_turns):
        a_i, chosen, _scores = agent.ensemble_act(history, rng)
        a_j, source = meta.act()
        a_i_actual, a_j_actual, r_i, r_j = env.step(a_i, a_j)
        meta.observe(a_j_actual, a_i_actual)
        history.append(turn_vector(a_i_actual, a_j_actual))
        pairs.append((source, chosen))
        total_i += r_i
        total_j += r_j

    turns = config.eval_turns
    return {"pairs": pairs, "mean_score_carla": total_i / turns, "mean_score_opponent": total_j / turns}
