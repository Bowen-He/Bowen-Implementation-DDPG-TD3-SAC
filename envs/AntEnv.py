from typing import Optional
import numpy as np

from gym import utils
from gym.envs.mujoco.ant_v4 import AntEnv

from gym.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class MyAnt(AntEnv):
    def __init__(self, 
                 xml_file="/home/bowen/Desktop/projects_Beta/Action-Space-Factorization/envs/assets/ant.xml", 
                 ctrl_cost_weight=0.5, 
                 use_contact_forces=False, 
                 contact_cost_weight=0.0005, 
                 healthy_reward=1, 
                 terminate_when_unhealthy=True, 
                 healthy_z_range=(0.1, 1.0), 
                 contact_force_range=(-1.0, 1.0), 
                 reset_noise_scale=0.1, 
                 exclude_current_positions_from_observation=True, 
                 **kwargs
        ):
        super().__init__(xml_file, ctrl_cost_weight, use_contact_forces, contact_cost_weight, healthy_reward, terminate_when_unhealthy, healthy_z_range, contact_force_range, reset_noise_scale, exclude_current_positions_from_observation, **kwargs)
        self._step_count = 0
        print("Using My Ant")
    
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._step_count = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, _, info = super().step(action)
        xy_velocity = abs(info["x_velocity"]) + abs(info["y_velocity"])
        self._step_count += 1
        if self._step_count == 1000:
            terminated = True
        return observation, xy_velocity, terminated, False, info
    
env = MyAnt()