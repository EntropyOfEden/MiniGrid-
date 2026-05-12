from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .networks import MiniGridCNN
from .replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        device: torch.device,
        lr: float = 1e-4,
        gamma: float = 0.99,
        use_double_dqn: bool = False,
        replay_capacity: int = 100_000,
    ):
        self.device = device
        self.gamma = gamma
        self.n_actions = n_actions
        self.use_double_dqn = use_double_dqn
        c, h, w = obs_shape
        self.policy_net = MiniGridCNN(c, n_actions).to(device)
        self.target_net = copy.deepcopy(self.policy_net).to(device)
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(replay_capacity, obs_shape, n_actions, device)

    def hard_update_target(self) -> None:
        """Hard sync target ← online（与 SB3 `tau=1` 的 polyak 更新等价）。"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act_epsilon_greedy(self, obs, epsilon: float) -> int:
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.n_actions, (1,)).item()
        with torch.no_grad():
            q = self.policy_net(obs.unsqueeze(0))
            return int(q.argmax(dim=1).item())

    def push(self, obs, action, reward, next_obs, terminated: bool):
        self.buffer.push(obs, action, reward, next_obs, terminated)

    def learn(self, batch_size: int):
        if len(self.buffer) < batch_size:
            return None
        obs, actions, rewards, next_obs, terminals = self.buffer.sample(batch_size)

        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.policy_net(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_obs).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_obs).max(dim=1).values
            target = rewards + (1.0 - terminals) * self.gamma * next_q

        q = self.policy_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(q, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()

    def state_dict(self):
        return {
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        self.policy_net.load_state_dict(state["policy"])
        self.target_net.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
