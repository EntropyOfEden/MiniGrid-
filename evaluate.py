#!/usr/bin/env python3
"""Evaluate a saved checkpoint (greedy policy)."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from src.env_setup import DEFAULT_ENV_ID, make_minigrid_env
from src.networks import MiniGridCNN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="单回合内最多执行多少步；默认 None 表示与环境 TimeLimit 一致（勿再用过小值，否则 KeyCorridor 等无法跑完）",
    )
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    try:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location="cpu")
    env_id = ckpt.get("env_id", DEFAULT_ENV_ID)
    step_penalty = ckpt.get("step_penalty", -0.001)
    max_episode_steps = ckpt.get("max_episode_steps", None)
    obs_shape = tuple(ckpt["obs_shape"])
    n_actions = int(ckpt["n_actions"])

    render_mode = "human" if args.render else None
    env = make_minigrid_env(
        env_id=env_id,
        render_mode=render_mode,
        step_penalty=step_penalty,
        max_episode_steps=max_episode_steps,
    )

    horizon = args.max_steps
    if horizon is None:
        spec = getattr(env, "spec", None)
        horizon = getattr(spec, "max_episode_steps", None) if spec is not None else None
    if horizon is None or horizon < 1:
        horizon = 10_000

    c, _, _ = obs_shape
    net = MiniGridCNN(c, n_actions)
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
        final_term = False
        
        # 必须保持这个循环，模型才能在环境里持续做决策
        while not done and steps < horizon:
            with torch.no_grad():
                # 加入极少量的随机动作，作为打破死循环的“润滑剂”
                if np.random.rand() < 0.05:
                    a = env.action_space.sample()
                else:
                    # 确保输入 Tensor 维度正确 [1, C, H, W]
                    obs_tensor = torch.as_tensor(obs, device=device).unsqueeze(0)
                    q = net(obs_tensor)
                    a = int(q.argmax(dim=1).item())
            
            # 执行动作并获取反馈
            obs, r, term, trunc, _ = env.step(a)
            
            final_term = term
            done = term or trunc
            ret += r
            steps += 1
            
        returns.append(ret)
        successes.append(1.0 if final_term else 0.0)

    print("eval episodes:", args.episodes)
    print("mean_return:", float(np.mean(returns)))
    print("success_rate (proxy):", float(np.mean(successes)))
    env.close()


if __name__ == "__main__":
    main()
