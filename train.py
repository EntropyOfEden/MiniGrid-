#!/usr/bin/env python3
"""Train DQN / Double-DQN on MiniGrid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.dqn_agent import DQNAgent
from src.env_setup import make_minigrid_env


def epsilon_by_step(step: int, eps_start: float, eps_end: float, eps_decay_steps: int) -> float:
    if step >= eps_decay_steps:
        return eps_end
    frac = step / float(eps_decay_steps)
    return eps_start + frac * (eps_end - eps_start)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    env = make_minigrid_env(env_id=args.env_id, step_penalty=args.step_penalty)
    obs, _ = env.reset()
    obs_shape = tuple(obs.shape)
    n_actions = int(env.action_space.n)

    agent = DQNAgent(
        obs_shape=obs_shape,
        n_actions=n_actions,
        device=device,
        lr=args.lr,
        gamma=args.gamma,
        use_double_dqn=args.double_dqn,
        replay_capacity=args.replay_capacity,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)

    global_step = 0
    episode_returns = []
    episode_lengths = []
    losses = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0
        while not done:
            eps = epsilon_by_step(global_step, args.eps_start, args.eps_end, args.eps_decay_steps)
            obs_t = torch.as_tensor(obs, device=device)
            action = agent.act_epsilon_greedy(obs_t, eps)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.push(obs, action, reward, next_obs, done)

            loss = agent.learn(args.batch_size)
            if loss is not None:
                losses.append(loss)

            if global_step % args.target_update_every == 0:
                agent.soft_copy_target()

            obs = next_obs
            ep_ret += reward
            ep_len += 1
            global_step += 1

            if global_step >= args.max_steps:
                done = True

        episode_returns.append(ep_ret)
        episode_lengths.append(ep_len)

        row = {
            "episode": ep,
            "global_step": global_step,
            "return": ep_ret,
            "length": ep_len,
            "epsilon": eps,
            "double_dqn": bool(args.double_dqn),
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        if (ep + 1) % args.log_every == 0:
            tail = episode_returns[-args.log_every :]
            print(
                f"ep {ep + 1}/{args.episodes}  "
                f"steps={global_step}  "
                f"ret_mean_{args.log_every}={np.mean(tail):.3f}  "
                f"eps={eps:.3f}  "
                f"buf={len(agent.buffer)}"
            )

        if global_step >= args.max_steps:
            break

    ckpt_path = out_dir / "agent.pt"
    torch.save(
        {
            "policy_state_dict": agent.policy_net.state_dict(),
            "obs_shape": obs_shape,
            "n_actions": n_actions,
            "use_double_dqn": bool(args.double_dqn),
            "env_id": args.env_id,
            "step_penalty": float(args.step_penalty),
        },
        ckpt_path,
    )
    print("saved:", ckpt_path)

    env.close()
    return episode_returns, episode_lengths, losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=200_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--replay-capacity", type=int, default=80_000)
    p.add_argument("--target-update-every", type=int, default=1000)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=80_000)
    p.add_argument("--step-penalty", type=float, default=-0.001)
    p.add_argument("--double-dqn", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--output-dir", type=str, default="runs/default")
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    train(args)


if __name__ == "__main__":
    main()
