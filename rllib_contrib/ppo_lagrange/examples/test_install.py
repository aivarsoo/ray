import ray
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.tune.logger import pretty_print


env = "Pendulum-v1"
stop_conditions = {"sampler_results/episode_reward_mean": 175}


num_gpus = 1
ray.init(num_gpus=num_gpus, local_mode=True)

algo = (PPOConfig()
        .training(
        train_batch_size=1024,
        vf_clip_param=10.0,
        lambda_=0.1,
        gamma=0.95,
        lr=3e-4,     
        sgd_minibatch_size=128,
        model={"fcnet_activation": "relu"},
        )
    .framework("torch")
    .environment(env=env)
    .resources(num_gpus=num_gpus)
    .rollouts(
        enable_connectors=True,
        observation_filter='MeanStdFilter',
    )
    .build())

print("Starting training")
for i in range(1000):
    result = algo.train()
    print(pretty_print(result))
print("done")


