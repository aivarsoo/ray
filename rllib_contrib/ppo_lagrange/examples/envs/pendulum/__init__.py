from gym.envs import register
from ray.tune.registry import register_env

from .pendulum import pendulum_cfg
from .pendulum import PendulumEnv
from .pendulum import SafePendulumEnv

# register(
#     id='CustomPendulum-v0',
#     entry_point='pendulum:PendulumEnv',
#     max_episode_steps=pendulum_cfg['max_ep_len']
# )

# register(
#     id='Safeendulum-v0',
#     entry_point='pendulum:SafePendulumEnv',
#     max_episode_steps=pendulum_cfg['max_ep_len']
# )


def custom_pendulum_creator(env_config):
    return PendulumEnv()  # return an env instance


register_env("CustomPendulum-v0", custom_pendulum_creator)


def safe_pendulum_creator(env_config):
    return SafePendulumEnv()  # return an env instance


register_env("SafePendulum-v0", safe_pendulum_creator)
