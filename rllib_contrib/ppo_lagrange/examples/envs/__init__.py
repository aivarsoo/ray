
import safety_gymnasium
from .saute_env import SauteEnv

from ray.tune.registry import register_env

from envs.pendulum.pendulum import PendulumEnv
from envs.pendulum.pendulum import SafePendulumEnv
from typing import Dict


def custom_pendulum_creator(env_config: Dict):
    return PendulumEnv() 

def safe_pendulum_creator(env_config: Dict):
    return SafePendulumEnv()  

def safegym_env_creator(env_name:str):    
    return safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safety_gymnasium.make(env_name))

def saute_safety_gym(env_name:str,config:Dict):
    env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safety_gymnasium.make(env_name))
    return SauteEnv(env, safety_budget=config['cost_lim'], 
        saute_discount_factor=config['cost_gamma'],
        max_ep_len=config['max_ep_len'],
        use_reward_shaping=False,
        use_state_augmentation=True)
    
def saute_pendulum_creator(env_config: Dict):
    env = SafePendulumEnv() 
    return SauteEnv(env, 
        safety_budget=env_config["cost_lim"], 
        saute_discount_factor=env_config["cost_gamma"],
        max_ep_len=env_config["max_ep_len"],
        use_reward_shaping=False,
        use_state_augmentation=True)

register_env("CustomPendulum-v0", custom_pendulum_creator)

register_env("PointGoal1-v0", lambda config: safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safety_gymnasium.make("SafetyPointGoal1-v0")))

register_env("SautePointGoal1-v0", lambda config: saute_safety_gym("SafetyPointGoal1-v0"))

register_env("SafePendulum-v0", safe_pendulum_creator)

register_env("SautePendulum-v0", saute_pendulum_creator)