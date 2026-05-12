from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], n_actions: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        # 仅当 Gymnasium terminated=True（MDP 真终止）时为 1；truncated 超时仍为 0，以便 TD 仍 bootstrap next_q
        self.terminals = np.zeros(self.capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def push(self, obs, action, reward, next_obs, terminated: bool):
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_obs[self.idx] = next_obs
        self.terminals[self.idx] = float(terminated)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=self.device),
            torch.as_tensor(self.actions[idx], device=self.device),
            torch.as_tensor(self.rewards[idx], device=self.device),
            torch.as_tensor(self.next_obs[idx], device=self.device),
            torch.as_tensor(self.terminals[idx], device=self.device),
        )

    def __len__(self):
        return self.size
