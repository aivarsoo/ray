"""
Proximal Policy Optimization (PPO)
==================================

This file defines the distributed Algorithm class for proximal policy
optimization.
See `ppo_[tf|torch]_policy.py` for the definition of the policy loss.

Detailed documentation: https://docs.ray.io/en/master/rllib-algorithms.html#ppo
"""
from collections import deque
import dataclasses
import logging
from typing import (
    Dict,
    Optional,
    Type,
    TYPE_CHECKING,
    Union
)

import numpy as np

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.algorithm_config import NotProvided
from ray.rllib.algorithms.ppo.ppo import PPO
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_learner import LEARNER_RESULTS_KL_KEY
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.execution.rollout_ops import standardize_fields
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import multi_gpu_train_one_step
from ray.rllib.execution.train_ops import train_one_step
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import ALL_MODULES
from ray.rllib.utils.metrics import NUM_AGENT_STEPS_SAMPLED
from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED
from ray.rllib.utils.metrics import SAMPLE_TIMER
from ray.rllib.utils.metrics import SYNCH_WORKER_WEIGHTS_TIMER
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import ResultDict
from ray.util.debug import log_once

from ppo_lagrange.callbacks import ComputeEpisodeCostCallback
from ppo_lagrange.cost_postprocessing import CostValuePostprocessing
from ppo_lagrange.ppo_catalog import PPOLagrangeCatalog
from ppo_lagrange.ppo_learner import (
    P_PART, 
    I_PART, 
    D_PART, 
    MEAN_CONSTRAINT_VIOL, 
    PPOLagrangeLearnerHyperparameters
    )
from ppo_lagrange.utils import substract_average

if TYPE_CHECKING:
    from ray.rllib.core.learner.learner import Learner


logger = logging.getLogger(__name__)


class PPOLagrangeConfig(PPOConfig):
    """Defines a configuration class from which a PPO Lagrange Algorithm can be built.

    Example:
        >>> from ppo_lagrange import PPOLagrangeConfig
        >>> config = PPOLagrangeConfig()  # doctest: +SKIP
        >>> config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3)  # doctest: +SKIP
        >>> config = config.resources(num_gpus=0)  # doctest: +SKIP
        >>> config = config.rollouts(num_rollout_workers=4)  # doctest: +SKIP
        >>> print(config.to_dict())  # doctest: +SKIP
        >>> # Build a Algorithm object from the config and run 1 training iteration.
        >>> algo = config.build(env="CartPole-v1")  # doctest: +SKIP
        >>> algo.train()  # doctest: +SKIP

    Example:
        >>> from ppo_lagrange import PPOLagrangeConfig
        >>> from ray import air
        >>> from ray import tune
        >>> config = PPOLagrangeConfig()
        >>> # Print out some default values.
        >>> print(config.clip_param)  # doctest: +SKIP
        >>> # Update the config object.
        >>> config.training(  # doctest: +SKIP
        ... lr=tune.grid_search([0.001, 0.0001]), clip_param=0.2
        ... )
        >>> # Set the config object's env.
        >>> config = config.environment(env="CartPole-v1")   # doctest: +SKIP
        >>> # Use to_dict() to get the old-style python config dict
        >>> # when running with tune.
        >>> tune.Tuner(  # doctest: +SKIP
        ...     "PPOLagrange",
        ...     run_config=air.RunConfig(stop={"episode_reward_mean": 200}),
        ...     param_space=config.to_dict(),
        ... ).fit()
    """

    def __init__(self, algo_class=None):
        """Initializes a PPOConfig instance."""
        super().__init__(algo_class=algo_class or PPOLagrange)

        self.use_cost_critic = True
        self.use_cost_gae = True
        self.cvf_clip_param = 100.0
        self.cvf_loss_coeff = 1.0
        self.cost_advant_std = False
        self.clip_cost_cvf = False
        self.cost_lambda_ = 1.0
        self.cost_gamma = 1.0
        self.cost_limit = 25.0
        self.init_penalty_coeff = 0.05
        self.penalty_coeff_config = {
            'learn_penalty_coeff': False,
            'penalty_coeff_lr': 1e-2,
            'pid_coeff': {P_PART: 0.0, I_PART:1e-2,  D_PART: 0.0},
            'polyak_coeff': 0.1,
            'max_penalty_coeff': 100}

        self.callbacks_class = ComputeEpisodeCostCallback
        self.keep_per_episode_custom_metrics = True

    @override(PPOConfig)
    def get_default_rl_module_spec(self) -> SingleAgentRLModuleSpec:
        if self.framework_str == "torch":
            from ppo_lagrange.torch.ppo_torch_rl_module import (
                PPOLagrangeTorchRLModule,
            )

            return SingleAgentRLModuleSpec(
                module_class=PPOLagrangeTorchRLModule, catalog_class=PPOLagrangeCatalog
            )
        else:
            raise ValueError(
                f"The framework {self.framework_str} is not supported. "
                "Use 'torch'"
            )

    @override(PPOConfig)
    def get_default_learner_class(self) -> Union[Type["Learner"], str]:
        if self.framework_str == "torch":
            from ppo_lagrange.torch.ppo_torch_learner import (
                PPOLagrangeTorchLearner,
            )

            return PPOLagrangeTorchLearner

        else:
            raise ValueError(
                f"The framework {self.framework_str} is not supported. "
                "Use 'torch'."
            )

    @override(PPOConfig)
    def get_learner_hyperparameters(self) -> PPOLagrangeLearnerHyperparameters:
        base_hps = super().get_learner_hyperparameters()
        return PPOLagrangeLearnerHyperparameters(
            cost_advant_std=self.cost_advant_std,
            clip_cost_cvf=self.clip_cost_cvf,
            use_cost_critic=self.use_cost_critic,
            cvf_loss_coeff=self.cvf_loss_coeff,
            cvf_clip_param=self.cvf_clip_param,
            cost_limit=self.cost_limit,
            penalty_coeff_config=self.penalty_coeff_config,
            penalty_coefficient=self.init_penalty_coeff,
            smoothed_violation=0.0,
            i_part=self.init_penalty_coeff,
            **dataclasses.asdict(base_hps),
        )

    @override(PPOConfig)
    def training(
        self,
        *,
        cost_advant_std: Optional[bool] = NotProvided,
        clip_cost_cvf: Optional[bool] = NotProvided,
        use_cost_critic: Optional[bool] = NotProvided,
        use_cost_gae: Optional[bool] = NotProvided,
        cost_lambda_: Optional[float] = NotProvided,
        cost_gamma: Optional[float] = NotProvided,
        cvf_loss_coeff: Optional[float] = NotProvided,
        cvf_clip_param: Optional[float] = NotProvided,
        cost_limit: Optional[float] = NotProvided,
        init_penalty_coeff: Optional[float] = NotProvided,
        penalty_coeff_config: Optional[Dict] = NotProvided,
        **kwargs,
    ) -> "PPOLagrangeConfig":
        """Sets the training related configuration.

        Args:
            use_cost_critic: Should use a critic for the cost as a baseline (otherwise don't use value
                baseline; required for using GAE).
            use_cost_gae: If true, use the Generalized Advantage Estimator (GAE)
                with a cost value function, see https://arxiv.org/pdf/1506.02438.pdf.
            cost_lambda_: The GAE (lambda) parameter for the cost GAE.
            cvf_loss_coeff: Coefficient of the cost value function loss. IMPORTANT: you must
                tune this if you set vf_share_layers=True inside your model's config.
            cvf_clip_param: Clip param for the cost value function. Note that this is
                sensitive to the scale of the rewards. If your expected V is large,
                increase this.
            init_penalty_coeff: Initial Lagrange penalty coefficient.
            cost_limit: Cost limit in the constraint.
            penalty_coeff_config: Configuration dictionary for penalty coefficient learning.
        Returns:
            This updated AlgorithmConfig object.
        """

        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        # TODO (sven): Move to generic AlgorithmConfig.
        if use_cost_critic is not NotProvided:
            self.use_cost_critic = use_cost_critic
            # TODO (Kourosh) This is experimental. Set learner_hps parameters as
            # well. Don't forget to remove .use_critic from algorithm config.
        if use_cost_gae is not NotProvided:
            self.use_cost_gae = use_cost_gae
        if clip_cost_cvf is not NotProvided:
            self.clip_cost_cvf = clip_cost_cvf
        if cost_advant_std is not NotProvided:
            self.cost_advant_std = cost_advant_std
        if cost_lambda_ is not NotProvided:
            self.cost_lambda_ = cost_lambda_
        if cost_gamma is not NotProvided:
            self.cost_gamma = cost_gamma
        if cvf_loss_coeff is not NotProvided:
            self.cvf_loss_coeff = cvf_loss_coeff
        if cvf_clip_param is not NotProvided:
            self.cvf_clip_param = cvf_clip_param
        if cost_limit is not NotProvided:
            self.cost_limit = cost_limit
        if init_penalty_coeff is not NotProvided:
            self.init_penalty_coeff = float(init_penalty_coeff)
        if penalty_coeff_config is not NotProvided:
            self.penalty_coeff_config = penalty_coeff_config
        return self

    @override(PPOConfig)
    def validate(self) -> None:
        # Call super's validation method.
        super().validate()
                
              
        assert self.penalty_coeff_config['penalty_coeff_lr'] > 0, "Learning rate must be positive"
        assert 1.0 >= self.penalty_coeff_config['polyak_coeff'] > 0, "Polyak coefficient must be between zero and one"
        assert self.penalty_coeff_config['max_penalty_coeff'] > 0, "Maximum penalty coefficient must be positive"

        for value in self.penalty_coeff_config['pid_coeff'].values():
            assert value >= 0, "PID coefficients must be nonnegative"
        
        # assert self.penalty_coeff_config['pid_coeff'][I_PART] > 0, "Integral gain coefficients must be positive"


            
        
class PPOLagrange(PPO):
    @classmethod
    @override(PPO)
    def get_default_config(cls) -> AlgorithmConfig:
        return PPOLagrangeConfig()

    @classmethod
    @override(PPO)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        if config["framework"] == "torch":

            from ppo_lagrange.ppo_torch_policy import PPOLagrangeTorchPolicy
            return PPOLagrangeTorchPolicy

        else:
            raise ValueError(
                f"The framework {config['framework']} is not supported. "
                "Use 'torch'"
            )

    @ExperimentalAPI
    def training_step(self) -> ResultDict:
        # Collect SampleBatches from sample workers until we have a full batch.
        with self._timers[SAMPLE_TIMER]:
            if self.config.count_steps_by == "agent_steps":
                train_batch = synchronous_parallel_sample(
                    worker_set=self.workers,
                    max_agent_steps=self.config.train_batch_size,
                )
            else:
                train_batch = synchronous_parallel_sample(
                    worker_set=self.workers, max_env_steps=self.config.train_batch_size
                )

        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # Standardize advantages
        train_batch = standardize_fields(train_batch, ["advantages"])
        if self.config.cost_advant_std:
            train_batch = standardize_fields(train_batch, ["cost_advantages"])
        else:
            train_batch = substract_average(train_batch, ["cost_advantages"])
        
        # Train
        if self.config._enable_learner_api:
            # TODO (Kourosh) Clearly define what train_batch_size
            #  vs. sgd_minibatch_size and num_sgd_iter is in the config.
            # TODO (Kourosh) Do this inside the Learner so that we don't have to do
            #  this back and forth communication between driver and the remote
            #  learner actors.
            is_module_trainable = self.workers.local_worker().is_policy_to_train
            self.learner_group.set_is_module_trainable(is_module_trainable)
            train_results = self.learner_group.update(
                train_batch,
                minibatch_size=self.config.sgd_minibatch_size,
                num_iters=self.config.num_sgd_iter,
            )

        elif self.config.simple_optimizer:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        if self.config._enable_learner_api:
            # The train results's loss keys are pids to their loss values. But we also
            # return a total_loss key at the same level as the pid keys. So we need to
            # subtract that to get the total set of pids to update.
            # TODO (Kourosh): We should also not be using train_results as a message
            #  passing medium to infer which policies to update. We could use
            #  policies_to_train variable that is given by the user to infer this.
            policies_to_update = set(train_results.keys()) - {ALL_MODULES}
        else:
            policies_to_update = list(train_results.keys())

        # TODO (Kourosh): num_grad_updates per each policy should be accessible via
        # train_results
        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
            "num_grad_updates_per_policy": {
                pid: self.workers.local_worker().policy_map[pid].num_grad_updates
                for pid in policies_to_update
            },
        }

        # Update weights - after learning on the local worker - on all remote
        # workers.
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            if self.workers.num_remote_workers() > 0:
                from_worker_or_learner_group = None
                if self.config._enable_learner_api:
                    # sync weights from learner_group to all rollout workers
                    from_worker_or_learner_group = self.learner_group
                self.workers.sync_weights(
                    from_worker_or_learner_group=from_worker_or_learner_group,
                    policies=policies_to_update,
                    global_vars=global_vars,
                )
            elif self.config._enable_learner_api:
                weights = self.learner_group.get_weights()
                self.workers.local_worker().set_weights(weights)

        if self.config._enable_learner_api:

            kl_dict = {}
            if self.config.use_kl_loss:
                for pid in policies_to_update:
                    kl = train_results[pid][LEARNER_RESULTS_KL_KEY]
                    kl_dict[pid] = kl
                    if np.isnan(kl):
                        logger.warning(
                            f"KL divergence for Module {pid} is non-finite, this will "
                            "likely destabilize your model and the training process. "
                            "Action(s) in a specific state have near-zero probability. "
                            "This can happen naturally in deterministic environments "
                            "where the optimal policy has zero mass for a specific "
                            "action. To fix this issue, consider setting `kl_coeff` to "
                            "0.0 or increasing `entropy_coeff` in your config."
                        )
            lp_dict = {}
            if self.config.penalty_coeff_config['learn_penalty_coeff']:
                for pid in policies_to_update:
                    # cost_error = train_results[pid][MEAN_CONSTRAINT_VIOL]
                    acc_costs = train_batch[pid][CostValuePostprocessing.RETURNS]
                    lp_dict[pid] = acc_costs
                    if np.isnan(acc_costs).any():
                        logger.warning(
                            f"Largrangian coefficient for Module {pid} is non-finite"
                            # "likely destabilize your model and the training process. "
                            # "Action(s) in a specific state have near-zero probability. "
                            # "This can happen naturally in deterministic environments "
                            # "where the optimal policy has zero mass for a specific "
                            # "action. To fix this issue, consider setting `kl_coeff` to "
                            # "0.0 or increasing `entropy_coeff` in your config."
                        )
            # triggers a special update method on RLOptimizer to update the KL values.
            additional_results = self.learner_group.additional_update(
                module_ids_to_update=policies_to_update,
                sampled_kl_values=kl_dict,
                sampled_lp_values=lp_dict,
                timestep=self._counters[NUM_AGENT_STEPS_SAMPLED],
            )
            for pid, res in additional_results.items():
                train_results[pid].update(res)

            return train_results

        # For each policy: Update KL scale and warn about possible issues
        for policy_id, policy_info in train_results.items():
            # Update KL loss with dynamic scaling
            # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            self.get_policy(policy_id).update_kl(kl_divergence)

            # Warn about excessively high value function loss
            scaled_vf_loss = (
                self.config.vf_loss_coeff * policy_info[LEARNER_STATS_KEY]["vf_loss"]
            )
            policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
            if (
                log_once("ppo_warned_lr_ratio")
                and self.config.get("model", {}).get("vf_share_layers")
                and scaled_vf_loss > 100
            ):
                logger.warning(
                    "The magnitude of your value function loss for policy: {} is "
                    "extremely large ({}) compared to the policy loss ({}). This "
                    "can prevent the policy from learning. Consider scaling down "
                    "the VF loss by reducing vf_loss_coeff, or disabling "
                    "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss)
                )
            # Warn about bad clipping configs.
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
            mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
            if (
                log_once("ppo_warned_vf_clip")
                and mean_reward > self.config.vf_clip_param
            ):
                self.warned_vf_clip = True
                logger.warning(
                    f"The mean reward returned from the environment is {mean_reward}"
                    f" but the vf_clip_param is set to {self.config['vf_clip_param']}."
                    f" Consider increasing it for policy: {policy_id} to improve"
                    " value function convergence."
                )

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results
