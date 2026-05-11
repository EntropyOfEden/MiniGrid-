import sys
from pathlib import Path

import gymnasium as gym
import minigrid

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_setup import DEFAULT_ENV_ID

ENV_ID = DEFAULT_ENV_ID  # 与 train 默认一致；也可改成任意已注册的 MiniGrid ID

env = gym.make(ENV_ID, render_mode="human")
obs, info = env.reset()
print(ENV_ID, "obs:", type(obs), getattr(obs, "shape", None))

for t in range(300):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc:
        obs, info = env.reset()

env.close()
