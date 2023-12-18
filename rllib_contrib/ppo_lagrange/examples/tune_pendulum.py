import pprint

from ray.rllib.algorithms.ppo.ppo import PPO
from ppo_lagrange import PPOLagrange

import ray
from ray import air, tune
import envs.pendulum

if __name__ == "__main__":

    max_concurrent_trials, num_samples, num_gpus, num_cpus = 1, 1, 1, 2
    ray.init(num_cpus=num_cpus,num_gpus=num_gpus, local_mode=True) # num_gpus=num_gpus, 
    stop = {"iterations_since_restore": 300}    
    params = {
                "num_workers": num_cpus-1,
                "framework": "torch",
                "vf_clip_param": 10000.0,                
                "enable_connectors": True,
                "model": {"fcnet_activation": "relu"},
                "env": "SafePendulum-v0",
                "gamma": 0.95,
                # tunable parameters
                "train_batch_size": 200,
                "clip_param": 0.2,
                "sgd_minibatch_size":20, #128, #tune.choice([128, 256]),
                "lr": 3e-3,#tune.choice([1e-4, 3e-4, 1e-3, 3e-3]),
                "lambda": 0.95,#tune.choice([0.95, 0.97]),          
                "num_sgd_iter": 80,# tune.choice([15, 30]),
                #### safety parameters
                "cost_limit": 20.0,
                "cost_lambda_": 0.95,
                "cost_gamma": 0.99,
                "cvf_clip_param": 10000.0,
                "init_penalty_coeff": 0.5,
                "penalty_coeff_config": {
                    'learn_penalty_coeff': True,
                    'penalty_coeff_lr': 1e-3,
                    # 'pid_coeff': {"P":  tune.choice([5e-1, 1.0, 5.0]), "I": 1.0, "D": tune.choice([5e-1, 1.0])},
                    'pid_coeff': {"P":  5.0, "I": 1.0, "D": 0},
                    'polyak_coeff': 0.1, 
                    'max_penalty_coeff': 100.0
                    },
                "seed": 42,
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
        run_config=air.RunConfig(stop=stop),
    )
    results = tuner.fit()
