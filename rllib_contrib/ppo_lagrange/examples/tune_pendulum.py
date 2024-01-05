import pprint

from ray.rllib.algorithms.ppo.ppo import PPO
from ppo_lagrange import PPOLagrange

import ray
from ray import air, tune
import envs.pendulum
from typing import Any, Callable, Collection, Dict, List, Optional, Tuple, Union

       
if __name__ == "__main__":
    max_concurrent_trials, num_samples, num_gpus, num_cpus = 1, 1, 1, 2
    ray.init(num_cpus=num_cpus,num_gpus=num_gpus, local_mode=True)
    stop = {"iterations_since_restore": 200}    
    params = {
                "num_workers": num_cpus-1,
                "framework": "torch",
                "vf_clip_param": 10000.0,        
                "enable_connectors": True,
                "model": {"vf_share_layers": False, "fcnet_activation": "relu"},
                "env": "SafePendulum-v0",
                # "env": "Pendulum-v1",
                "gamma": 0.95,
                # tunable parameters
                "train_batch_size": 2000,
                "clip_param": 0.2,
                # "sgd_minibatch_size": 200,
                "lr": 1e-3,
                "lambda": 0.95,       
                "num_sgd_iter": 10,
                #### safety parameters
                'learn_penalty_coeff': True,
                "cost_lambda_": 0.95,
                "cost_gamma": 0.99,
                "safety_config" : {
                    "cost_limit": 20.0,
                    "cvf_clip_param": 10000.0,
                    "init_penalty_coeff": 0.5,
                    'penalty_coeff_lr': 1e-3,  
                    'max_penalty_coeff': 100.0,
                    "p_coeff": 0.0,                
                    "d_coeff": 0.0,
                    "aw_coeff": 0.0,
                },
                "seed": 42,
    }
    tuner = tune.Tuner(
        PPOLagrange,
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            # metric="custom_metrics/episode_cost_mean",
            # mode="min",
            scheduler=None,
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials,
        ),
        param_space=params,
        run_config=air.RunConfig(stop=stop)
    )
    results = tuner.fit()
    print("done")
