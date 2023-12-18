import gym 
from gymnasium import Wrapper
import numpy as np
from typing import List, Tuple


class SafetyGymnasiumWrapper(Wrapper):
    """
    Base class for the safety gym environments 
    """

    def __init__(
            self,
            env: gym.Env,             
    ):
        self.env = env        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def seed(self, seed: int = None) -> List[int]:
        return self.env.seed(seed)

    def step(self, action: np.ndarray) -> Tuple:
        obs, reward, done, info = self.env.step(action)
        info['pos_com'] = self.env.world.robot_com()  # saving position of the robot to plot
        return obs, reward, done, False, info

    def reset(self) -> Tuple:
        obs = self.env.reset()
        return obs, {"cost": 0, "pos_com": self.env.world.robot_com()}


