"""Ring buffer of (window, a_i, a_j_next) triples for opponent-model training.

Mirrors Algorithm 2 lines 2/6/7: an empty buffer is initialized once per
strategy, each training iteration's rollout is added to it, and a random
minibatch is sampled from it (not just the latest rollout) to train phi_j.
"""
from __future__ import annotations

import random
from collections import deque

import torch


class OpponentReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.windows: deque = deque(maxlen=capacity)
        self.a_i: deque = deque(maxlen=capacity)
        self.a_j_next: deque = deque(maxlen=capacity)

    def push_batch(self, windows: torch.Tensor, a_i: torch.Tensor, a_j_next: torch.Tensor) -> None:
        """windows: (N, T, 4); a_i, a_j_next: (N,) long tensors."""
        for i in range(windows.shape[0]):
            self.windows.append(windows[i])
            self.a_i.append(a_i[i])
            self.a_j_next.append(a_j_next[i])

    def __len__(self) -> int:
        return len(self.windows)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = len(self)
        idx = random.sample(range(n), k=min(batch_size, n))
        windows = torch.stack([self.windows[i] for i in idx])
        a_i = torch.stack([self.a_i[i] for i in idx])
        a_j_next = torch.stack([self.a_j_next[i] for i in idx])
        return windows, a_i, a_j_next
