# MiniGrid 深度强化学习导航（DQN / Double DQN）

本项目实现基于 **深度 Q 网络（DQN）** 与 **Double DQN** 的 MiniGrid 迷宫导航智能体，依赖 [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) 与 [Minigrid](https://github.com/Farama-Foundation/Minigrid)（均为 MIT License）。实现参考了经典 DQN（Mnih et al., 2015）与 Double DQN（van Hasselt et al., 2016）的公式与工程惯例，代码为本仓库独立撰写。

## 环境

- Python 3.10+
- 建议使用虚拟环境：`python -m venv .venv`，激活后执行 `pip install -r requirements.txt`

若在 Windows 上遇到 `torch` / `c10.dll` 加载失败，可尝试重装 CPU 版：

`pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu`

## 快速验证（随机策略）

在项目根目录执行：

`python scripts/run_random_demo.py`

## 训练

`python train.py --output-dir runs/exp1`

启用 Double DQN：

`python train.py --double-dqn --output-dir runs/exp2`

默认训练环境为 **`MiniGrid-ObstructedMaze-2Q-v1`**（门锁、钥匙与阻挡球的迷宫类任务，详见 MiniGrid 文档）。若要换回简单空房间可加上 `--env-id MiniGrid-Empty-8x8-v0`。该任务较难，若收敛慢可适当增大 `--max-steps`、`--episodes`。

常用参数：`--env-id`、`--episodes`、`--max-steps`、`--eps-decay-steps`、`--step-penalty`。

训练过程将写入 `metrics.jsonl`，并在结束时保存 `agent.pt`。

## 对比实验与曲线

快速预算（用于调试）：

`python scripts/compare_double_dqn.py --quick`

默认预算会更长；额外参数会透传给 `train.py`，例如：

`python scripts/compare_double_dqn.py --env-id MiniGrid-FourRooms-v0`

对比结果图：`runs/compare_dqn_double.png`。

## 评估 / 演示录屏

`python evaluate.py --ckpt runs/exp1/agent.pt --episodes 30`

开启可视化窗口（便于 OBS 录屏）：

`python evaluate.py --ckpt runs/exp1/agent.pt --episodes 10 --render`

## 目录说明

| 路径 | 说明 |
|------|------|
| `src/env_setup.py` | `DEFAULT_ENV_ID`、MiniGrid 封装、全局可视图像观测、每步惩罚奖励整形 |
| `src/networks.py` | 卷积 Q 网络 |
| `src/replay_buffer.py` | 经验回放 |
| `src/dqn_agent.py` | DQN / Double DQN 更新逻辑 |
| `train.py` | 训练入口 |
| `evaluate.py` | 贪心策略评估 |
| `scripts/compare_double_dqn.py` | 基线与改进对比绘图 |
| `作品说明.md` | 大作业说明文档（功能、算法训练、分工） |
