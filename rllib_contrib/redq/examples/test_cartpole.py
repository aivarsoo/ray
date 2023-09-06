import ray
import yaml
from ray import air
from ray import tune
from ray.rllib.algorithms.sac.sac import SACConfig
from ray.tune.logger import pretty_print
from redq import REDQConfig

config_file_path = 'rllib/tuned_examples/sac/cartpole-sac.yaml'
with open(config_file_path, 'r') as file:
    config_file = yaml.safe_load(file)
    config_file = [*config_file.values()][0]

algo = 'redq'
# algo = 'sac'
if algo == 'redq':
    config_cls = REDQConfig
else:
    config_cls = SACConfig
env = config_file['env']  # CartPole-v1
algo = config_file['run']  # PPO
stop_conditions = config_file['stop']
# stop:
#     sampler_results/episode_reward_mean: 150
#     timesteps_total: 100000

num_gpus = 1
ray.init(num_gpus=num_gpus)

algo = (config_cls()
        .training(
            gamma=config_file['config']['gamma'],
            train_batch_size=config_file['config']['train_batch_size'],
            tau=config_file['config']['tau'],
            target_network_update_freq=config_file['config']['target_network_update_freq'],
            optimization_config=config_file['config']['optimization'],
            # training_intensity=1,
            # num_steps_sampled_before_learning_starts=10000,
)
    .framework(
            config_file['config']['framework'])
    .environment(
            env=env)
    .resources(
            num_gpus=num_gpus)
    .build())

for i in range(100000):
    result = algo.train()
    print(pretty_print(result))
print("done")
