import logging
from typing import Dict
from typing import List
from typing import Type
from typing import Union

import ppo_lagrange
from ppo_lagrange.cost_postprocessing import compute_gae_for_sample_batch
from ppo_lagrange.cost_postprocessing import CostValuePostprocessing
from ppo_lagrange.cost_postprocessing import Postprocessing
from ppo_lagrange.cost_postprocessing import RewardValuePostprocessing
from ppo_lagrange.utils import validate_config
from ppo_lagrange.utils import CostAndValueNetworkMixins
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import EntropyCoeffSchedule
from ray.rllib.policy.torch_mixins import KLCoeffMixin
from ray.rllib.policy.torch_mixins import LearningRateSchedule
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import apply_grad_clipping
from ray.rllib.utils.torch_utils import explained_variance
from ray.rllib.utils.torch_utils import sequence_mask
from ray.rllib.utils.torch_utils import warn_if_infinite_kl_divergence
from ray.rllib.utils.typing import TensorType


torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class PPOLagrangeTorchPolicy(
    CostAndValueNetworkMixins,
    LearningRateSchedule,
    EntropyCoeffSchedule,
    KLCoeffMixin,
    TorchPolicyV2,
):
    def __init__(self, observation_space, action_space, config):
        config = dict(ppo_lagrange.PPOLagrangeConfig().to_dict(), **config)
        validate_config(config)

        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        CostAndValueNetworkMixins.__init__(self, config)
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        KLCoeffMixin.__init__(self, config)

        self._initialize_loss_from_dummy_batch()

    @override(TorchPolicyV2)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for Proximal Policy Objective with a constraint.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO Lagrangian loss tensor given the input batch.
        """

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            # TODO smorad: should we do anything besides warn? Could discard KL term
            # for this update
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        def update_value_function(current_batch: SampleBatch, post_processing: Postprocessing, use_critic: bool, clip_param: float):
            surrogate_loss = torch.min(
                current_batch[post_processing.ADVANTAGES] * logp_ratio,
                current_batch[post_processing.ADVANTAGES]
                * torch.clamp(
                    logp_ratio, 1 -
                    self.config["clip_param"], 1 + self.config["clip_param"]
                ),
            )

            # Compute a value function loss.
            if use_critic:
                value_fn_out = model.value_function()
                vf_loss = torch.pow(
                    value_fn_out - current_batch[post_processing.VALUE_TARGETS], 2.0
                )
                vf_loss_clipped = torch.clamp(vf_loss, 0, clip_param)
                mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
            # Ignore the value function.
            else:
                value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
                vf_loss_clipped = mean_vf_loss = torch.tensor(
                    0.0).to(surrogate_loss.device)
            return surrogate_loss, vf_loss_clipped, mean_vf_loss, value_fn_out

        surrogate_loss, vf_loss_clipped, mean_vf_loss, value_fn_out = update_value_function(
            train_batch,
            RewardValuePostprocessing,
            self.config['use_critic'],
            self.config["vf_clip_param"]
        )

        cost_surrogate_loss, cvf_loss_clipped, mean_cvf_loss, cost_value_fn_out = update_value_function(
            train_batch,
            RewardValuePostprocessing,
            self.config['use_critic'],
            self.config["cvf_clip_param"]
        )

        surrogate_loss -= cost_surrogate_loss

        total_loss = reduce_mean_valid(
            - surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            + self.config["cvf_loss_coeff"] * cvf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[RewardValuePostprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_cvf_loss"] = mean_cvf_loss
        model.tower_stats["cvf_explained_var"] = explained_variance(
            train_batch[CostValuePostprocessing.VALUE_TARGETS], cost_value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return total_loss

    # TODO: Make this an event-style subscription (e.g.:
    #  "after_gradients_computed").
    @override(TorchPolicyV2)
    def extra_grad_process(self, local_optimizer, loss):
        return apply_grad_clipping(self, local_optimizer, loss)

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy(
            {
                "cur_kl_coeff": self.kl_coeff,
                "cur_lr": self.cur_lr,
                "total_loss": torch.mean(
                    torch.stack(self.get_tower_stats("total_loss"))
                ),
                "policy_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_policy_loss"))
                ),
                "vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_vf_loss"))
                ),
                "vf_explained_var": torch.mean(
                    torch.stack(self.get_tower_stats("vf_explained_var"))
                ),
                "cvf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_cvf_loss"))
                ),
                "cvf_explained_var": torch.mean(
                    torch.stack(self.get_tower_stats("cvf_explained_var"))
                ),
                "kl": torch.mean(torch.stack(self.get_tower_stats("mean_kl_loss"))),
                "entropy": torch.mean(
                    torch.stack(self.get_tower_stats("mean_entropy"))
                ),
                "entropy_coeff": self.entropy_coeff,
            }
        )

    @override(TorchPolicyV2)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        # Do all post-processing always with no_grad().
        # Not using this here will introduce a memory leak
        # in torch (issue #6962).
        # TODO: no_grad still necessary?

        with torch.no_grad():
            return compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )
