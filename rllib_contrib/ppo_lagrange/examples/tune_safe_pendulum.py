import pprint

from ray.rllib.algorithms.ppo.ppo import PPO
from ppo_lagrange import PPOLagrange

import ray
from ray import air, tune
import envs 
from ray.tune import CLIReporter

if __name__ == "__main__":

    # # Limit the number of rows.
    reporter = CLIReporter(max_progress_rows=10)
    # Add a custom metric column, in addition to the default metrics.
    # Note that this must be a metric that is returned in your training results.
    reporter.add_metric_column("episode_cost")
    reporter.add_metric_column("custom_metrics/episode_cost_mean")
    
    max_concurrent_trials, num_samples, num_gpus, num_cpus = 1, 1, 1, 5
    ray.init(num_cpus=num_cpus,num_gpus=num_gpus, local_mode=True) # num_gpus=num_gpus, 
    stop = {"iterations_since_restore": 200}    
    cost_lim = 20.0
    max_ep_len = 200
    cost_gamma = 0.95
    params = {
                "num_rollout_workers": num_cpus-1,
                "framework": "torch",
                "vf_clip_param": 10000.0,                
                "keep_per_episode_custom_metrics": True,
                "metrics_num_episodes_for_smoothing": 20,
                "observation_filter": "MeanStdFilter",
                "enable_connectors": True,
                "model": {"vf_share_layers": False, "fcnet_activation": "relu"},
                "env": "SafePendulum-v0",
                # "env": "SautePendulum-v0",
                "env_config":dict(
                    cost_lim=cost_lim,
                    max_ep_len=max_ep_len,
                    cost_gamma=cost_gamma
                ),
                "gamma": 0.95,
                # tunable parameters
                "train_batch_size": 4000,
                "clip_param": 0.2,                
                "sgd_minibatch_size": 4000, 
                "lr": 1e-3,
                "lambda": 0.95,
                "num_sgd_iter": 80,
                #### safety parameters
                "cost_advant_std": False,
                "clip_cost_cvf": False, 
                "cost_limit": cost_lim,
                "cost_lambda_": 0.97,
                "cost_gamma": cost_gamma,
                "cvf_clip_param": 10000.0,
                "init_penalty_coeff": -1.0,
                "penalty_coeff_config": {
                    'learn_penalty_coeff': True,
                    'penalty_coeff_lr': 1e-2,
                    'pid_coeff': {"P":  1e-1, "D": 0}, 
                    'polyak_coeff': 0.2, 
                    'max_penalty_coeff': 100.0
                    },
                "seed": tune.choice([42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]),
    }
    tuner = tune.Tuner(
        PPOLagrange,
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            scheduler=None,
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials,
        ),
        param_space=params,
        run_config=air.RunConfig(stop=stop, progress_reporter=reporter),
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