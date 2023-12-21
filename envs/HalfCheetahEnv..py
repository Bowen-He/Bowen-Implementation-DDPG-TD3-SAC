import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box

from gym.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class MyHalfCheetah(HalfCheetahEnv):
    def __init__(self, 
                 forward_reward_weight=1, 
                 ctrl_cost_weight=0.1, 
                 reset_noise_scale=0.1, 
                 exclude_current_positions_from_observation=True, 
                 **kwargs):
        super().__init__(forward_reward_weight, ctrl_cost_weight, reset_noise_scale, exclude_current_positions_from_observation, **kwargs)
    