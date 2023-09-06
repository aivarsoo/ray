import ray
import yaml
from ray import air
from ray import tune
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.tune.logger import pretty_print

config_file_path = 'rllib/tuned_examples/ppo/cartpole-ppo.yaml'
with open(config_file_path, 'r') as file:
    config_file = yaml.safe_load(file)
    config_file = [*config_file.values()][0]


env = config_file['env']  # CartPole-v1
algo = config_file['run']  # PPO
stop_conditions = config_file['stop']
# stop:
#     sampler_results/episode_reward_mean: 150
#     timesteps_total: 100000

num_gpus = 1
ray.init(num_gpus=num_gpus)

algo = (PPOConfig()
        .training(
            gamma=config_file['config']['gamma'],
            lr=config_file['config']['lr'],
            num_sgd_iter=config_file['config']['num_sgd_iter'],
            vf_loss_coeff=config_file['config']['vf_loss_coeff'],
            model=config_file['config']['model'],
)
    .framework(
            config_file['config']['framework']
)
    .environment(
            env=env
)
    .resources(
            num_gpus=num_gpus
)
    # .rollouts(
    # num_rollout_workers=config_file['config']['num_workers'],
    # enable_connectors=config_file['config']['enable_connectors'],
    # observation_filter=config_file['config']['observation_filter'],
    # )
    .build())

for i in range(100):
    result = algo.train()
    print(pretty_print(result))
print("done")
