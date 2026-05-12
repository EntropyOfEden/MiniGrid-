#!/usr/bin/env python3
"""Train DQN / Double-DQN on MiniGrid.

训练循环与超参命名对齐 Stable Baselines3 的 DQN 实现（见
https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
及 stable_baselines3/dqn/dqn.py 默认参数），便于对照网上教程与 RL Zoo。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.dqn_agent import DQNAgent
from src.env_setup import DEFAULT_ENV_ID, make_minigrid_env


def linear_schedule(progress_remaining: float, initial_value: float, final_value: float, end_fraction: float) -> float:
    """与 SB3 `common.utils.LinearSchedule` 相同：progress_remaining 从 1（起点）→ 0（训练预算用尽）。"""
    fraction = 1.0 - progress_remaining
    if fraction > end_fraction:
        return final_value
    return initial_value + (fraction / end_fraction) * (final_value - initial_value)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    env = make_minigrid_env(
        env_id=args.env_id,
        step_penalty=args.step_penalty,
        max_episode_steps=args.max_episode_steps,
    )
    obs, _ = env.reset()
    obs_shape = tuple(obs.shape)
    n_actions = int(env.action_space.n)

    agent = DQNAgent(
        obs_shape=obs_shape,
        n_actions=n_actions,
        device=device,
        lr=args.learning_rate,
        gamma=args.gamma,
        use_double_dqn=args.double_dqn,
        replay_capacity=args.buffer_size,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)

    global_step = 0
    episode_returns = []
    episode_lengths = []
    episode_terminated = []
    losses = []

    total_budget = max(int(args.max_steps), 1)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0
        last_term = False
        last_trunc = False
        while not done:
            # 与 SB3 一致：用「剩余训练进度」调度 ε（总步数上限为 total_budget）
            p_done = min(1.0, global_step / float(total_budget))
            progress_remaining = max(0.0, 1.0 - p_done)
            eps = linear_schedule(
                progress_remaining,
                args.exploration_initial_eps,
                args.exploration_final_eps,
                args.exploration_fraction,
            )

            obs_t = torch.as_tensor(obs, device=device)
            action = agent.act_epsilon_greedy(obs_t, eps)
            next_obs, reward, term, trunc, _ = env.step(action)
            last_term, last_trunc = term, trunc
            done = term or trunc
            agent.push(obs, action, reward, next_obs, term)

            global_step += 1

            # SB3: num_timesteps > learning_starts 且按 train_freq 触发梯度
            if global_step > args.learning_starts and global_step % args.train_freq == 0:
                for _ in range(args.gradient_steps):
                    loss = agent.learn(args.batch_size)
                    if loss is not None:
                        losses.append(loss)

            # SB3: target_update_interval 环境步（单环境 n_envs=1）
            if global_step > 0 and global_step % args.target_update_interval == 0:
                agent.hard_update_target()

            obs = next_obs
            ep_ret += reward
            ep_len += 1

            if global_step >= args.max_steps:
                done = True

        episode_returns.append(ep_ret)
        episode_lengths.append(ep_len)
        episode_terminated.append(1.0 if last_term else 0.0)

        p_done = min(1.0, global_step / float(total_budget))
        progress_remaining = max(0.0, 1.0 - p_done)
        eps_log = linear_schedule(
            progress_remaining,
            args.exploration_initial_eps,
            args.exploration_final_eps,
            args.exploration_fraction,
        )

        row = {
            "episode": ep,
            "global_step": global_step,
            "return": ep_ret,
            "length": ep_len,
            "terminated": last_term,
            "truncated": last_trunc,
            "task_success": bool(last_term),
            "epsilon": eps_log,
            "double_dqn": bool(args.double_dqn),
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        if (ep + 1) % args.log_every == 0:
            tail = episode_returns[-args.log_every :]
            tail_term = episode_terminated[-args.log_every :]
            term_rate = float(np.mean(tail_term))
            print(
                f"ep {ep + 1}/{args.episodes}  "
                f"steps={global_step}  "
                f"ret_mean_{args.log_every}={np.mean(tail):.3f}  "
                f"term_rate_{args.log_every}={term_rate:.3f}  "
                f"eps={eps_log:.3f}  "
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
            "max_episode_steps": args.max_episode_steps,
        },
        ckpt_path,
    )
    print("saved:", ckpt_path)

    env.close()
    return episode_returns, episode_lengths, losses


def main():
    # 默认与 SB3 DQN 一致（stable_baselines3/dqn/dqn.py）
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID)
    p.add_argument("--episodes", type=int, default=15000)
    p.add_argument("--max-steps", type=int, default=3_000_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--learning-starts", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-freq", type=int, default=4)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--target-update-interval", type=int, default=1000)
    p.add_argument("--exploration-fraction", type=float, default=0.5)
    p.add_argument("--exploration-initial-eps", type=float, default=1.0)
    p.add_argument("--exploration-final-eps", type=float, default=0.05)
    p.add_argument("--step-penalty", type=float, default=-0.003)
    p.add_argument("--double-dqn", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--output-dir", type=str, default="runs/vanilla_dqn-DoorKey-8x8-toward-6")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-episode-steps", type=int, default=None)
    args = p.parse_args()

    train(args)


if __name__ == "__main__":
    main()
