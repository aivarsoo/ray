import pprint

from ray.rllib.algorithms.ppo.ppo import PPO
from ppo_lagrange import PPOLagrange

import ray
from ray import air, tune
import envs 

import os
os.environ['TUNE_PLACEMENT_GROUP_AUTO_DISABLED'] = '1'

if __name__ == "__main__":

    max_concurrent_trials, num_samples, num_gpus, num_cpus = 1, 1, 1, 2
    ray.init(num_gpus=num_gpus, num_cpus=num_cpus, local_mode=True)
    stop = {
        "num_agent_steps_sampled": 2e+6,
        # 'training_iteration': 1
        }    
    cost_lim = 25.0
    max_ep_len = 1000
    cost_gamma = 0.995
    params = {
                "num_rollout_workers": num_cpus-1,
                "framework": "torch",
                "vf_clip_param": 10000.0,                
                "metrics_num_episodes_for_smoothing": 20,
                "observation_filter": "MeanStdFilter",
                "enable_connectors": True,
                "model": {"vf_share_layers": False, "fcnet_activation": "relu", "fcnet_hiddens": [256, 256, 256]},
                "env": "SafetyPointGoal1-v0",
                "env_config":dict(
                    cost_lim=cost_lim,
                    max_ep_len=max_ep_len,
                    cost_gamma=cost_gamma
                ),
                "gamma": 0.995,
                # tunable parameters
                "train_batch_size": 30000,
                "clip_param": 0.2,
                "sgd_minibatch_size": 30000,  
                "lr": 1e-4, 
                "lambda": 0.97,
                "num_sgd_iter": 5, 
                #### safety parameters
                'learn_penalty_coeff': True,
                "cost_lambda_": 0.97,
                "cost_gamma": cost_gamma,
                "safety_config" : {
                    "cost_limit": cost_lim,
                    "cvf_clip_param": 10000.0,
                    "init_penalty_coeff": 0.3,
                    'polyak_coeff': 1.0,                     
                    'penalty_coeff_lr': 5e-3,  
                    'max_penalty_coeff': 100.0,
                    "p_coeff": 0.0,                
                    "d_coeff": 0.0,
                    "aw_coeff": 0.0,
                },      
                # "safety":{                
                # "cost_limit": cost_lim,
                # "cost_advant_std": False, #tune.choice([False]),
                # "clip_cost_cvf": False, #tune.choice([False]), 
                # "cost_lambda_": 0.97,
                # "cost_gamma": cost_gamma,
                # "cvf_clip_param": 10000.0,                
                # "init_penalty_coeff": -0.5,
                # "penalty_coeff_config": {
                    # 'learn_penalty_coeff': False,
                    # 'penalty_coeff_lr': tune.grid_search([1e-3, 5e-3, 1e-2]),
                    # 'pid_coeff': {"P":  0, "D": 0},
                    # 'polyak_coeff': 1.0, 
                    # 'penalty_coeff_lr': tune.choice([6e-3, 1e-2]),
                    # 'pid_coeff': {"P": 0.0, "D": 0.0},
                    # 'polyak_coeff':  tune.choice([0.1, 0.2]), 
                    # 'max_penalty_coeff': 100.0
                    # },
                # }
                "seed": 44,
    }
    # for env in ["SautePointGoal1-v0", "PointGoal1-v0"]:
    #   params['env'] = env
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