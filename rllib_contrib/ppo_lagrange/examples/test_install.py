import ray
from envs.pendulum import pendulum_cfg
from envs.pendulum import PendulumEnv
from ppo_lagrange import PPOLagrangeConfig
from ppo_lagrange.callbacks import ComputeEpisodeCostCallback
from ray.tune.logger import pretty_print


env = "SafePendulum-v0"
stop_conditions = {"sampler_results/episode_reward_mean": 175}


num_gpus = 1
ray.init(num_gpus=num_gpus, local_mode=True)

algo = (PPOLagrangeConfig()
        .training(
            gamma=0.99,
            lambda_=0.97,
            cost_gamma=0.99,
            cost_lambda_=0.97,
            cost_limit=30.0,
            penalty_coeff_config={
                "learn_penalty_coeff": True,
                "pid_coeff": {"P": 0.0, "I": 1.0, "D": 0.0},
                'pid_polyak_coeff': {"P": 0.05, "I": 1.0, "D": 0.05},
                'max_penalty_coeff': 100
            },
            lr=1e-3,
            clip_param=0.2,
            train_batch_size=1024,
            num_sgd_iter=80,
            model={"fcnet_activation": "relu"},
)
    .framework(
    "torch"
)
    .environment(
            env=env,
)
    .resources(
            num_gpus=num_gpus
)
    .rollouts(
            enable_connectors=True,
            observation_filter='MeanStdFilter',
)
    .callbacks(ComputeEpisodeCostCallback)
    .reporting(keep_per_episode_custom_metrics=True)
    # .evaluation(
    #     # evaluation_num_workers=1,
    #     evaluation_interval=1,
    #     evaluation_duration=1,
    #     evaluation_duration_unit = "episodes"
    # )
    .build())
print("Starting training")
for i in range(100):
    result = algo.train()
    print(pretty_print(result))
print("done")
