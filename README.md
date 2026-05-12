# MiniGrid 深度强化学习导航（DQN / Double DQN）

本项目在 [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) + [Minigrid](https://github.com/Farama-Foundation/Minigrid) 上实现 **DQN** 与 **Double DQN** 迷宫导航智能体。算法与工程习惯参考经典 DQN（Mnih et al., 2015）与 Double DQN（van Hasselt et al., 2016），代码为本仓库独立撰写。

## 环境与依赖

- Python 3.10+
- 建议：`python -m venv .venv`，激活后 `pip install -r requirements.txt`

若在 Windows 上遇到 `torch` / `c10.dll` 加载失败，可尝试重装 CPU 版：

`pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu`

## 快速体验（随机策略）

在项目根目录：

`python scripts/run_random_demo.py`

## 训练

`python train.py --output-dir runs/exp1`

启用 Double DQN：

`python train.py --double-dqn --output-dir runs/exp2`

默认任务由 `src/env_setup.py` 中的 **`DEFAULT_ENV_ID`** 决定（当前为 **`MiniGrid-DoorKey-8x8-v0`**：捡钥匙 → 开门 → 到达目标）。

**难度阶梯（同一套代码，改 `--env-id` 即可）**：`MiniGrid-Empty-8x8-v0` → `MiniGrid-DoorKey-5x5-v0` → `MiniGrid-DoorKey-6x6-v0` → 默认 `MiniGrid-DoorKey-8x8-v0` → `MiniGrid-KeyCorridorS3R1-v0` / `MiniGrid-KeyCorridorS5R3-v0` → 更难如 `MiniGrid-ObstructedMaze-2Q-v1`。收敛慢时可增大 `--max-steps`、`--episodes`。

常用参数：`--env-id`、`--episodes`、`--max-steps`、`--learning-starts`、`--train-freq`、`--buffer-size`、`--target-update-interval`、`--exploration-fraction`、`--exploration-initial-eps`、`--exploration-final-eps`、`--gradient-steps`、`--learning-rate`、`--batch-size`、`--step-penalty`、`--max-episode-steps`（命名与语义对齐 [Stable Baselines3 DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)）。

### `MiniGrid-KeyCorridorS5R3-v0` 参考（较难对照）

| 项 | 值 |
|----|-----|
| `gym.spec(...).kwargs` | `room_size=5`，`num_rows=3` |
| 底层 `KeyCorridorEnv.max_steps` | **750**（超时 `truncated=True`） |
| 未包装地图规模 | `width=13`，`height=13`（随 minigrid 版本可能略有差异） |

训练初期多见 `truncated`、少见 `terminated` 属正常。日志中 **`term_rate_N`** 表示最近 N 个 episode 内任务完成比例；`metrics.jsonl` 含 `task_success`、`terminated`、`truncated`。

**TD 目标（Gymnasium 语义）**：仅当 **`terminated`** 时在目标中截断 bootstrap；因步数上限导致的 **`truncated`** 仍对 `next_obs` 做 bootstrap，与常见实现一致。

训练输出：`metrics.jsonl`、`agent.pt`（含 `env_id`、`step_penalty`、`max_episode_steps` 等，供评估对齐）。

## 对比实验

快速调试：

`python scripts/compare_double_dqn.py --quick`

完整对比（可透传 `train.py` 参数）：

`python scripts/compare_double_dqn.py --env-id MiniGrid-FourRooms-v0`

曲线输出：`runs/compare_dqn_double.png`。

## 评估与录屏演示

`python evaluate.py --ckpt runs/exp1/agent.pt --episodes 30`

开启窗口（便于 OBS 等录屏）：

`python evaluate.py --ckpt runs/exp1/agent.pt --episodes 10 --render`

`evaluate.py` 默认 **`--max-steps` 为 None**，与检查点及底层环境一致（例如 DoorKey-8x8 常见约 **640** 步上限、DoorKey-5x5 多为 **250**，均随 minigrid 版本略有差异）。若把 `--max-steps` 设得过小，可能在完成任务前被截断，导致 **`success_rate` 偏低**。更细节见 MiniGrid 文档或 `gym.spec`。

## 文档与分工

| 文件 | 说明 |
|------|------|
| `作品说明.md` | 大作业：功能、算法、训练方式、改进点 |
| `视频拍摄指南.md` | 合作同学录制讲解视频的步骤与检查清单 |
| `runs/README.md` | `runs/` 下各实验目录与 `metrics.jsonl` 含义 |

## 代码结构

| 路径 | 说明 |
|------|------|
| `src/env_setup.py` | `DEFAULT_ENV_ID`、环境封装、图像观测、每步惩罚 |
| `src/networks.py` | 卷积 Q 网络 |
| `src/replay_buffer.py` | 经验回放 |
| `src/dqn_agent.py` | DQN / Double DQN 更新 |
| `train.py` | 训练入口 |
| `evaluate.py` | 贪心策略评估 |
| `scripts/compare_double_dqn.py` | DQN vs Double DQN 对比绘图 |
