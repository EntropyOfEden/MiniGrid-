#!/usr/bin/env python3
"""Train vanilla DQN vs Double DQN with the same budget and plot episode returns."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_returns(metrics_path: Path):
    episodes = []
    returns = []
    with metrics_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            episodes.append(row["episode"])
            returns.append(row["return"])
    return np.array(episodes), np.array(returns)


def smooth(y, window: int):
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="small budget for smoke experiments")
    args, passthrough = p.parse_known_args()

    root = Path(__file__).resolve().parents[1]
    train_py = root / "train.py"
    for sub in ("compare_vanilla", "compare_double"):
        d = root / "runs" / sub
        if d.exists():
            shutil.rmtree(d)

    common = [
        sys.executable,
        str(train_py),
        "--cpu",
        "--log-every",
        "5",
    ]
    if args.quick:
        common += [
            "--episodes",
            "80",
            "--max-steps",
            "25000",
            "--learning-starts",
            "100",
            "--train-freq",
            "4",
            "--buffer-size",
            "50000",
            "--target-update-interval",
            "500",
            "--exploration-fraction",
            "0.5",
        ]
    else:
        common += ["--episodes", "350", "--max-steps", "180000"]

    common += passthrough

    run_vanilla = common + ["--output-dir", str(root / "runs" / "compare_vanilla")]
    run_double = common + ["--double-dqn", "--output-dir", str(root / "runs" / "compare_double")]

    print("running:", " ".join(run_vanilla))
    subprocess.run(run_vanilla, cwd=str(root), check=True)
    print("running:", " ".join(run_double))
    subprocess.run(run_double, cwd=str(root), check=True)

    ev, rv = load_returns(root / "runs" / "compare_vanilla" / "metrics.jsonl")
    ed, rd = load_returns(root / "runs" / "compare_double" / "metrics.jsonl")

    plt.figure(figsize=(9, 5))
    w = min(15, len(rv))
    if w < 3:
        plt.plot(ev, rv, label="DQN")
        plt.plot(ed, rd, label="Double DQN")
    else:
        plt.plot(ev[w - 1 :], smooth(rv, w), label=f"DQN (smoothed {w})")
        plt.plot(ed[w - 1 :], smooth(rd, w), label=f"Double DQN (smoothed {w})")
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.title("MiniGrid navigation: DQN vs Double DQN")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_png = root / "runs" / "compare_dqn_double.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print("saved plot:", out_png)


if __name__ == "__main__":
    main()
