import pprint

from ray.rllib.algorithms.ppo.ppo import PPO

import ray
from ray import air, tune
from ppo_lagrange.callbacks import ComputeEpisodeCostCallback
# from envs.pendulum import SafePendulumEnv
import envs.pendulum

if __name__ == "__main__":

    max_concurrent_trials, num_samples, num_gpus = 1, 50, 1
    ray.init(num_gpus=num_gpus)
    stop = {"iterations_since_restore": 500}    
    params = {
                "framework": "torch",
                "vf_clip_param": 10.0,
                "callbacks_class": ComputeEpisodeCostCallback,
                "keep_per_episode_custom_metrics": True,
                "observation_filter": "MeanStdFilter",
                "enable_connectors": True,
                "model": {"fcnet_activation": "relu"},
                "env": "SafePendulum-v0",
                "gamma": 0.99,
                "lambda": 0.97,
                # tunable parameters
                "train_batch_size": tune.choice([512, 1024, 2048, 4096]),                
                "lr": tune.choice([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]),
                # "gamma": tune.choice([0.95, 0.99]),
                "sgd_minibatch_size": tune.choice([512, 1024, 2048, 4096]),
                "seed": tune.choice([42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]),
    }
    tuner = tune.Tuner(
        PPO,
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            scheduler=None,
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials,
        ),
        param_space=params,
        run_config=air.RunConfig(stop=stop),
    )
    results = tuner.fit()

    best_result = results.get_best_result()

    print("\nBest performing trial's final reported metrics:\n")

    metrics_to_print = [
        "episode_reward_mean",
        "episode_reward_max",
        "episode_reward_min",
        "episode_len_mean",
    ]
    pprint.pprint(
        {k: v for k, v in best_result.metrics.items() if k in metrics_to_print}
    )