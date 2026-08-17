"""Hyperparameter config: loaded from configs/*.yaml (§V-A + README deviations)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    gamma: float
    learning_rate: float
    noise_level: float

    alpha: float
    beta: float
    window_length: int
    critic_loss_coef: float
    entropy_coef: float

    train_batch_size: int
    total_iterations: int
    warmup_iters: int
    opponent_replay_capacity: int
    opponent_minibatch: int

    eval_turns: int
    eval_repeats: int

    seed: int


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(**data)
