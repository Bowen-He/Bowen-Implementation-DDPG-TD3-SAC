import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box

from gym.envs.mujoco.walker2d_v4 import Walker2dEnv

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

class MyWalker(Walker2dEnv):
    def __init__(self, 
                 forward_reward_weight=1, 
                 ctrl_cost_weight=0.001, 
                 healthy_reward=1, 
                 terminate_when_unhealthy=True, 
                 healthy_z_range=(0.8, 2.0), 
                 healthy_angle_range=(-1.0, 1.0), 
                 reset_noise_scale=0.005, 
                 exclude_current_positions_from_observation=True, 
                 **kwargs):
        super().__init__(forward_reward_weight, ctrl_cost_weight, healthy_reward, terminate_when_unhealthy, healthy_z_range, healthy_angle_range, reset_noise_scale, exclude_current_positions_from_observation, **kwargs)
    