"""MiniGrid environment wiring (Farama Minigrid, MIT license)."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import ObservationWrapper, RewardWrapper, ActionWrapper
import numpy as np
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

# 默认环境 ID
DEFAULT_ENV_ID = "MiniGrid-DoorKey-8x8-v0"

class InjectAgentPosWrapper(ObservationWrapper):
    """确保在 FullyObsWrapper 后的全局视野中，能看见智能体的位置和朝向"""
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):
        # 获取原始环境对象以读取 agent 状态
        env = self.unwrapped
        # obs['image'] 的 shape 通常是 (W, H, 3)
        image = obs["image"].copy()
        x, y = env.agent_pos
        
        # MiniGrid 协议：通道0=物体(10是Agent), 通道1=颜色(0是红), 通道2=朝向(0-3)
        image[x, y, 0] = 10
        image[x, y, 1] = 0
        image[x, y, 2] = (env.agent_dir + 1) * 50
        
        obs["image"] = image
        return obs

class RemoveDoneActionWrapper(ActionWrapper):
    """强制移除 MiniGrid 的动作 6 (done)，防止智能体为了逃避惩罚而自杀"""
    def __init__(self, env):
        super().__init__(env)
        # 将动作空间缩小为 6 个 (0-5: left, right, forward, pickup, drop, toggle)
        self.action_space = gym.spaces.Discrete(6)

    def action(self, action):
        # 由于神经网络现在只输出 0-5，这里直接返回即可
        return action

class StepPenaltyRewardWrapper(RewardWrapper):
    """每步微量扣分，建议配合 RemoveDoneActionWrapper 使用"""
    def __init__(self, env, step_penalty: float = -0.001):
        super().__init__(env)
        self.step_penalty = float(step_penalty)

    def reward(self, reward):
        return reward + self.step_penalty

class ImageFloatCHW(ObservationWrapper):
    """
    将 HWC uint8 图像转换为 float32 CHW。
    注意：针对 MiniGrid 符号输入，除以 10.0 比除以 255.0 能保留更多特征。
    """
    def observation(self, obs):
        # 这里使用 10.0 因为 MiniGrid 符号值的最大值通常是 10
        x = np.asarray(obs, dtype=np.float32) / 10.0
        return np.transpose(x, (2, 0, 1))

def make_minigrid_env(
    env_id: str = DEFAULT_ENV_ID,
    fully_observable: bool = True,
    step_penalty: float | None = -0.001,
    render_mode: str | None = None,
    max_episode_steps: int | None = None,
):
    make_kw: dict = {"render_mode": render_mode}
    if max_episode_steps is not None:
        make_kw["max_episode_steps"] = int(max_episode_steps)
    
    try:
        env = gym.make(env_id, **make_kw)
    except TypeError:
        make_kw.pop("max_episode_steps", None)
        env = gym.make(env_id, **make_kw)
        if max_episode_steps is not None:
            from gymnasium.wrappers import TimeLimit
            env = TimeLimit(env, max_episode_steps=int(max_episode_steps))

    # 1. 开启全局视野
    if fully_observable:
        env = FullyObsWrapper(env)
        # 2. 注入智能体位置（解决原地转圈问题）
        env = InjectAgentPosWrapper(env)
    
    # 3. 提取图像通道
    env = ImgObsWrapper(env)
    
    # 4. 规范化图像（使用 /10.0）
    env = ImageFloatCHW(env)
    
    # 5. 添加步数惩罚
    if step_penalty is not None:
        env = StepPenaltyRewardWrapper(env, step_penalty=step_penalty)
    
    # 6. 屏蔽自杀动作（解决自杀问题）
    env = RemoveDoneActionWrapper(env)
    
    return env