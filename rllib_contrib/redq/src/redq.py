import logging
from typing import List
from typing import Optional
from typing import Type

import numpy as np
import torch
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.algorithm_config import NotProvided
from ray.rllib.algorithms.dqn.dqn import DQN
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.framework import try_import_tfp
from ray.rllib.utils.typing import ResultDict
from redq_torch_policy import REDQTorchPolicy

tf1, tf, tfv = try_import_tf()
tfp = try_import_tfp()

logger = logging.getLogger(__name__)


class REDQConfig(SACConfig):
    """Defines a configuration class from which a REDQ Algorithm can be built. Subclass of SACConfig

    Example:
        >>> config = REDQConfig().training(gamma=0.9, lr=0.01)  # doctest: +SKIP
        >>> config = config.resources(num_gpus=0)  # doctest: +SKIP
        >>> config = config.rollouts(num_rollout_workers=4)  # doctest: +SKIP
        >>> print(config.to_dict())  # doctest: +SKIP
        >>> # Build a Algorithm object from the config and run 1 training iteration.
        >>> algo = config.build(env="CartPole-v1")  # doctest: +SKIP
        >>> algo.train()  # doctest: +SKIP
    """

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or REDQ)
        self.ensemble_size = 2
        self.num_critics = 2
        self.target_prediction = lambda x: torch.min(torch.stack(x), axis=0)[0]
        self.q_prediction = lambda x: torch.min(torch.stack(x), axis=0)[0]
        self.value_function_clipping = False
        self.value_function_clip_value = 10

    @override(SACConfig)
    def training(
        self,
        *,
        ensemble_size: Optional[int] = NotProvided,
        num_critics: Optional[int] = NotProvided,
        target_prediction: Optional[str] = NotProvided,
        q_prediction: Optional[str] = NotProvided,
        value_function_clipping: Optional[bool] = NotProvided,
        value_function_clip_value: Optional[float] = NotProvided,
        **kwargs,
    ) -> "REDQConfig":
        super().training(**kwargs)
        if ensemble_size is not NotProvided:
            self.ensemble_size = ensemble_size
        if target_prediction is not NotProvided and target_prediction != 'min':
            self.target_prediction = lambda x: getattr(
                torch, target_prediction)(torch.stack(x), axis=0)
        if q_prediction is not NotProvided and q_prediction != 'min':
            self.q_prediction = lambda x: getattr(
                torch, q_prediction)(torch.stack(x), axis=0)
        # if q_prediction is not NotProvided:
        #     self.q_prediction = q_prediction
        if value_function_clipping is not NotProvided:
            self.value_function_clipping = value_function_clipping
        if value_function_clip_value is not NotProvided:
            self.value_function_clip_value = value_function_clip_value
        if num_critics is not NotProvided:
            self.num_critics = num_critics
        return self

    @override(SACConfig)
    def validate(self) -> None:
        # Call super's validation method.
        super().validate()
        assert self.ensemble_size >= self.num_critics, "Number of critics should be smaller or equal to ensemble size"
        # assert type(
        #     self.target_prediction) is str, "Target predicion is a string, specifically, one of the following strings: 'min', 'mean' or 'median'"
        # assert self.target_prediction in [
        #     'min', 'mean', 'median'], "Target predicion can only be performed using 'min', 'mean' or 'median' functions"
        # assert type(
        #     self.q_prediction) is str, "Target predicion is a string, specifically, one of the following strings: 'min', 'mean' or 'median'"
        # assert self.q_prediction in [
        #     'min', 'mean', 'median'], "Target predicion can only be performed using 'min', 'mean' or 'median' functions"
        assert self.value_function_clip_value >= 0, "Values needs to be positive"

# def calculate_rr_weights(config: AlgorithmConfig) -> List[float]:
#     """Calculate the round robin weights for the rollout and train steps"""
#     if not config["training_intensity"]:
#         return [1, 1]

#     # Calculate the "native ratio" as:
#     # [train-batch-size] / [size of env-rolled-out sampled data]
#     # This is to set freshly rollout-collected data in relation to
#     # the data we pull from the replay buffer (which also contains old
#     # samples).
#     native_ratio = config["train_batch_size"] / (
#         config.get_rollout_fragment_length()
#         * config["num_envs_per_worker"]
#         # Add one to workers because the local
#         # worker usually collects experiences as well, and we avoid division by zero.
#         * max(config["num_workers"] + 1, 1)
#     )

#     # Training intensity is specified in terms of
#     # (steps_replayed / steps_sampled), so adjust for the native ratio.
#     sample_and_train_weight = config["training_intensity"] / native_ratio
#     if sample_and_train_weight < 1:
#         return [int(np.round(1 / sample_and_train_weight)), 1]
#     else:
#         return [1, int(np.round(sample_and_train_weight))]

# from ray.rllib.utils.metrics import (
#     NUM_ENV_STEPS_SAMPLED,
#     NUM_AGENT_STEPS_SAMPLED,
#     SAMPLE_TIMER,
# )
# from ray.rllib.utils.metrics import SYNCH_WORKER_WEIGHTS_TIMER
# from ray.rllib.execution.common import (
#     LAST_TARGET_UPDATE_TS,
#     NUM_TARGET_UPDATES,
# )
# from ray.rllib.utils.replay_buffers.utils import sample_min_n_steps_from_buffer
# from ray.rllib.execution.rollout_ops import (
#     synchronous_parallel_sample,
# )
# from ray.rllib.execution.train_ops import (
#     train_one_step,
#     multi_gpu_train_one_step,
# )
# from ray.rllib.utils.replay_buffers.utils import update_priorities_in_replay_buffer


class REDQ(DQN):
    "A REDQ implementation based on SAC."

    def __init__(self, *args, **kwargs):
        self._allow_unknown_subkeys += ["policy_model_config", "q_model_config"]
        super().__init__(*args, **kwargs)

    @classmethod
    @override(DQN)
    def get_default_config(cls) -> AlgorithmConfig:
        return REDQConfig()

    @classmethod
    @override(DQN)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        if config["framework"] == "torch":
            return REDQTorchPolicy
        else:
            raise NotImplementedError

    # @override(DQN)
    # def training_step(self) -> ResultDict:
    #     return super().training_step()

    # @override(DQN)
    # def training_step(self) -> ResultDict:
    #     """DQN training iteration function.

    #     Each training iteration, we:
    #     - Sample (MultiAgentBatch) from workers.
    #     - Store new samples in replay buffer.
    #     - Sample training batch (MultiAgentBatch) from replay buffer.
    #     - Learn on training batch.
    #     - Update remote workers' new policy weights.
    #     - Update target network every `target_network_update_freq` sample steps.
    #     - Return all collected metrics for the iteration.

    #     Returns:
    #         The results dict from executing the training iteration.
    #     """
    #     train_results = {}

    #     # We alternate between storing new samples and sampling and training
    #     store_weight, sample_and_train_weight = calculate_rr_weights(self.config)

    #     for _ in range(store_weight):
    #         # Sample (MultiAgentBatch) from workers.
    #         with self._timers[SAMPLE_TIMER]:
    #             new_sample_batch = synchronous_parallel_sample(
    #                 worker_set=self.workers, concat=True
    #             )

    #         # Update counters
    #         self._counters[NUM_AGENT_STEPS_SAMPLED] += new_sample_batch.agent_steps()
    #         self._counters[NUM_ENV_STEPS_SAMPLED] += new_sample_batch.env_steps()

    #         # Store new samples in replay buffer.
    #         self.local_replay_buffer.add(new_sample_batch)

    #     global_vars = {
    #         "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
    #     }

    #     # Update target network every `target_network_update_freq` sample steps.
    #     cur_ts = self._counters[
    #         NUM_AGENT_STEPS_SAMPLED
    #         if self.config.count_steps_by == "agent_steps"
    #         else NUM_ENV_STEPS_SAMPLED
    #     ]

    #     if cur_ts > self.config.num_steps_sampled_before_learning_starts:
    #         for _ in range(sample_and_train_weight):
    #             # Sample training batch (MultiAgentBatch) from replay buffer.
    #             train_batch = sample_min_n_steps_from_buffer(
    #                 self.local_replay_buffer,
    #                 self.config.train_batch_size,
    #                 count_by_agent_steps=self.config.count_steps_by == "agent_steps",
    #             )

    #             # Postprocess batch before we learn on it
    #             post_fn = self.config.get("before_learn_on_batch") or (lambda b, *a: b)
    #             train_batch = post_fn(train_batch, self.workers, self.config)

    #             # for policy_id, sample_batch in train_batch.policy_batches.items():
    #             #     print(len(sample_batch["obs"]))
    #             #     print(sample_batch.count)

    #             # Learn on training batch.
    #             # Use simple optimizer (only for multi-agent or tf-eager; all other
    #             # cases should use the multi-GPU optimizer, even if only using 1 GPU)
    #             if self.config.get("simple_optimizer") is True:
    #                 train_results = train_one_step(self, train_batch)
    #             else:
    #                 train_results = multi_gpu_train_one_step(self, train_batch)

    #             # Update replay buffer priorities.
    #             update_priorities_in_replay_buffer(
    #                 self.local_replay_buffer,
    #                 self.config,
    #                 train_batch,
    #                 train_results,
    #             )

    #             last_update = self._counters[LAST_TARGET_UPDATE_TS]
    #             if cur_ts - last_update >= self.config.target_network_update_freq:
    #                 to_update = self.workers.local_worker().get_policies_to_train()
    #                 self.workers.local_worker().foreach_policy_to_train(
    #                     lambda p, pid: pid in to_update and p.update_target()
    #                 )
    #                 self._counters[NUM_TARGET_UPDATES] += 1
    #                 self._counters[LAST_TARGET_UPDATE_TS] = cur_ts

    #             # Update weights and global_vars - after learning on the local worker -
    #             # on all remote workers.
    #             with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
    #                 self.workers.sync_weights(global_vars=global_vars)

    #     # Return all collected metrics for the iteration.
    #     return train_results


class _deprecated_default_config(dict):
    def __init__(self):
        super().__init__(REDQConfig().to_dict())

    @Deprecated(
        old="redq::DEFAULT_CONFIG",
        new="redq::REDQConfig(...)",
        error=True,
    )
    def __getitem__(self, item):
        return super().__getitem__(item)


DEFAULT_CONFIG = _deprecated_default_config()
