# `runs/` 目录说明

本目录存放训练输出：**每个实验一个子文件夹**，通常包含：

| 文件 | 含义 |
|------|------|
| `agent.pt` | 检查点（策略权重、`env_id`、观测形状等） |
| `metrics.jsonl` | 逐条训练指标（回报、`task_success`、`terminated`、`truncated` 等） |

## 当前示例实验

命名仅供参考，以你本地实际文件夹为准：

| 子目录（示例） | 算法 | 用途 |
|----------------|------|------|
| `double_dqn-DoorKey-8x8-toward-6` | Double DQN | 与 Vanilla 对照、讲改进效果 |
| `vanilla_dqn-DoorKey-8x8-toward-6` | Vanilla DQN | 基线，可与 Double DQN 对比曲线或录像 |

讲解或写报告时，可用 **`metrics.jsonl`** 画学习曲线，并用对应 **`agent.pt`** 运行 `evaluate.py` 录屏。对比脚本 `scripts/compare_double_dqn.py` 还会在 `runs/compare_dqn_double.png` 生成汇总图（路径在项目根 `runs/` 下）。
