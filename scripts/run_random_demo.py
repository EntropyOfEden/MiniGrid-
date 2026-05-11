#!/usr/bin/env python3
"""Smoke test: random actions on MiniGrid (open-source stack sanity check)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_setup import DEFAULT_ENV_ID, make_minigrid_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID)
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()

    env = make_minigrid_env(env_id=args.env_id, render_mode=None)
    obs, _ = env.reset()
    print("env_id:", args.env_id)
    print("obs shape:", obs.shape, "dtype:", obs.dtype)
    print("action_space:", env.action_space)

    rewards = []
    for t in range(args.steps):
        a = env.action_space.sample()
        obs, r, term, trunc, _ = env.step(a)
        rewards.append(r)
        if term or trunc:
            obs, _ = env.reset()
    print("random rollout: steps=", args.steps, "mean_reward=", float(np.mean(rewards)))
    env.close()


if __name__ == "__main__":
    main()
