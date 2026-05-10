"""MiniGrid environment wiring (Farama Minigrid, MIT license — see README attribution)."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import ObservationWrapper, RewardWrapper
import numpy as np
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper


class StepPenaltyRewardWrapper(RewardWrapper):
    """Small per-step penalty plus goal bonus (sparse MiniGrid reward stays)."""

    def __init__(self, env, step_penalty: float = -0.001):
        super().__init__(env)
        self.step_penalty = float(step_penalty)

    def reward(self, reward):
        return reward + self.step_penalty


class ImageFloatCHW(ObservationWrapper):
    """Convert HWC uint8 image to float32 CHW in [0, 1]."""

    def observation(self, obs):
        x = np.asarray(obs, dtype=np.float32) / 255.0
        return np.transpose(x, (2, 0, 1))


def make_minigrid_env(
    env_id: str = "MiniGrid-Empty-8x8-v0",
    fully_observable: bool = True,
    step_penalty: float | None = -0.001,
    render_mode: str | None = None,
):
    env = gym.make(env_id, render_mode=render_mode)
    if fully_observable:
        env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    env = ImageFloatCHW(env)
    if step_penalty is not None:
        env = StepPenaltyRewardWrapper(env, step_penalty=step_penalty)
    return env
