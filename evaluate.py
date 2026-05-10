#!/usr/bin/env python3
"""Evaluate a saved checkpoint (greedy policy)."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from src.env_setup import make_minigrid_env
from src.networks import NatureCNN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    try:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location="cpu")
    env_id = ckpt.get("env_id", "MiniGrid-Empty-8x8-v0")
    step_penalty = ckpt.get("step_penalty", -0.001)
    obs_shape = tuple(ckpt["obs_shape"])
    n_actions = int(ckpt["n_actions"])

    render_mode = "human" if args.render else None
    env = make_minigrid_env(env_id=env_id, render_mode=render_mode, step_penalty=step_penalty)

    c, _, _ = obs_shape
    net = NatureCNN(c, n_actions)
    net.load_state_dict(ckpt["policy_state_dict"])
    net.eval()

    device = torch.device("cpu")
    net.to(device)

    returns = []
    successes = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ret = 0.0
        steps = 0
        while not done and steps < args.max_steps:
            with torch.no_grad():
                q = net(torch.as_tensor(obs, device=device).unsqueeze(0))
                a = int(q.argmax(dim=1).item())
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ret += r
            steps += 1
        returns.append(ret)
        successes.append(1.0 if ret > 0.5 else 0.0)

    print("eval episodes:", args.episodes)
    print("mean_return:", float(np.mean(returns)))
    print("success_rate (proxy):", float(np.mean(successes)))
    env.close()


if __name__ == "__main__":
    main()
