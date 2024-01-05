import sys

import gym
from pendulum import pendulum_cfg
from pendulum import PendulumEnv
sys.path.append(".")


if __name__ == "__main__":
    from gym.envs import register

    register(
        id='CustomPendulum-v0',
        entry_point='pendulum:PendulumEnv',
        # max_episode_steps=pendulum_cfg['max_ep_len']
    )
    register(
        id='SafePendulum-v0',
        entry_point='pendulum:SafePendulumEnv',
        max_episode_steps=pendulum_cfg['max_ep_len']
    )
    env = gym.make('SafePendulum-v0', mode="deterministic")
    env.reset()
    states, actions, next_states, rewards, dones, infos = [
        env.reset()], [], [], [], [], []
    for _ in range(3000):
        a = env.action_space.sample()
        s, r, d, t, i = env.step(a)
        print(s)
        print(r)
        if t or d:
            print(d, t)
        states.append(s)
        actions.append(a)
        next_states.append(s)
        rewards.append(r)
        dones.append(d)
        infos.append(i)
    print("dones")
