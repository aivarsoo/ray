"""
PyTorch policy class used for REDQ.
"""
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import gymnasium as gym
import ray.experimental.tf_utils
import redq
import tree  # pip install dm_tree
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.algorithms.sac.sac_tf_policy import postprocess_trajectory
from ray.rllib.algorithms.sac.sac_tf_policy import validate_spaces
from ray.rllib.algorithms.sac.sac_torch_policy import _get_dist_class
from ray.rllib.algorithms.sac.sac_torch_policy import action_distribution_fn
from ray.rllib.algorithms.sac.sac_torch_policy import ComputeTDErrorMixin
from ray.rllib.algorithms.sac.sac_torch_policy import F
from ray.rllib.algorithms.sac.sac_torch_policy import setup_late_mixins
from ray.rllib.algorithms.sac.sac_torch_policy import stats
from ray.rllib.algorithms.sac.sac_torch_policy import TargetNetworkMixin
from ray.rllib.algorithms.sac.sac_torch_policy import torch
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchBeta
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian
from ray.rllib.models.torch.torch_action_dist import TorchDirichlet
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_action_dist import TorchSquashedGaussian
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.torch_utils import apply_grad_clipping
from ray.rllib.utils.torch_utils import concat_multi_gpu_td_errors
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.torch_utils import huber_loss
from ray.rllib.utils.typing import AlgorithmConfigDict
from ray.rllib.utils.typing import LocalOptimizer
from ray.rllib.utils.typing import ModelInputDict
from ray.rllib.utils.typing import TensorType
from utils import build_redq_model

# torch, nn = try_import_torch()
# F = nn.functional

logger = logging.getLogger(__name__)


# def _get_dist_class(
#     policy: Policy, config: AlgorithmConfigDict, action_space: gym.spaces.Space
# ) -> Type[TorchDistributionWrapper]:
#     """Helper function to return a dist class based on config and action space.

#     Args:
#         policy: The policy for which to return the action
#             dist class.
#         config: The Algorithm's config dict.
#         action_space (gym.spaces.Space): The action space used.

#     Returns:
#         Type[TFActionDistribution]: A TF distribution class.
#     """
#     if hasattr(policy, "dist_class") and policy.dist_class is not None:
#         return policy.dist_class
#     elif config["model"].get("custom_action_dist"):
#         action_dist_class, _ = ModelCatalog.get_action_dist(
#             action_space, config["model"], framework="torch"
#         )
#         return action_dist_class
#     elif isinstance(action_space, Discrete):
#         return TorchCategorical
#     elif isinstance(action_space, Simplex):
#         return TorchDirichlet
#     else:
#         assert isinstance(action_space, Box)
#         if config["normalize_actions"]:
#             return (
#                 TorchSquashedGaussian
#                 if not config["_use_beta_distribution"]
#                 else TorchBeta
#             )
#         else:
#             return TorchDiagGaussian


def build_redq_model_and_action_dist(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: AlgorithmConfigDict,
) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:
    """Constructs the necessary ModelV2 and action dist class for the Policy.

    Args:
        policy: The TFPolicy that will use the models.
        obs_space (gym.spaces.Space): The observation space.
        action_space (gym.spaces.Space): The action space.
        config: The SAC trainer's config dict.

    Returns:
        ModelV2: The ModelV2 to be used by the Policy. Note: An additional
            target model will be created in this function and assigned to
            `policy.target_model`.
    """
    model = build_redq_model(policy, obs_space, action_space, config)
    action_dist_class = _get_dist_class(policy, config, action_space)
    return model, action_dist_class


# def action_distribution_fn(
#     policy: Policy,
#     model: ModelV2,
#     input_dict: ModelInputDict,
#     *,
#     state_batches: Optional[List[TensorType]] = None,
#     seq_lens: Optional[TensorType] = None,
#     prev_action_batch: Optional[TensorType] = None,
#     prev_reward_batch=None,
#     explore: Optional[bool] = None,
#     timestep: Optional[int] = None,
#     is_training: Optional[bool] = None
# ) -> Tuple[TensorType, Type[TorchDistributionWrapper], List[TensorType]]:
#     """The action distribution function to be used the algorithm.

#     An action distribution function is used to customize the choice of action
#     distribution class and the resulting action distribution inputs (to
#     parameterize the distribution object).
#     After parameterizing the distribution, a `sample()` call
#     will be made on it to generate actions.

#     Args:
#         policy: The Policy being queried for actions and calling this
#             function.
#         model (TorchModelV2): The SAC specific model to use to generate the
#             distribution inputs (see sac_tf|torch_model.py). Must support the
#             `get_action_model_outputs` method.
#         input_dict: The input-dict to be used for the model
#             call.
#         state_batches (Optional[List[TensorType]]): The list of internal state
#             tensor batches.
#         seq_lens (Optional[TensorType]): The tensor of sequence lengths used
#             in RNNs.
#         prev_action_batch (Optional[TensorType]): Optional batch of prev
#             actions used by the model.
#         prev_reward_batch (Optional[TensorType]): Optional batch of prev
#             rewards used by the model.
#         explore (Optional[bool]): Whether to activate exploration or not. If
#             None, use value of `config.explore`.
#         timestep (Optional[int]): An optional timestep.
#         is_training (Optional[bool]): An optional is-training flag.

#     Returns:
#         Tuple[TensorType, Type[TorchDistributionWrapper], List[TensorType]]:
#             The dist inputs, dist class, and a list of internal state outputs
#             (in the RNN case).
#     """
#     # Get base-model output (w/o the SAC specific parts of the network).
#     model_out, _ = model(input_dict, [], None)
#     # Use the base output to get the policy outputs from the SAC model's
#     # policy components.
#     action_dist_inputs, _ = model.get_action_model_outputs(model_out)
#     # Get a distribution class to be used with the just calculated dist-inputs.
#     action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)
#     return action_dist_inputs, action_dist_class, []


def actor_critic_loss(
    policy: Policy,
    model: ModelV2,
    dist_class: Type[TorchDistributionWrapper],
    train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for the REDQ.

    Args:
        policy: The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[TorchDistributionWrapper]: The action distr. class.
        train_batch: The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    num_critics = policy.config['num_critics']
    target_predictor = policy.config['target_prediction']
    q_predictor = policy.config['q_prediction']
    # getattr(torch, policy.config['target_prediction'])
    # Look up the target model (tower) using the model tower.
    target_model = policy.target_models[model]

    # Should be True only for debugging purposes (e.g. test cases)!
    deterministic = policy.config["_deterministic_loss"]

    model_out_t, _ = model(
        SampleBatch(obs=train_batch[SampleBatch.CUR_OBS], _is_training=True), [], None
    )

    model_out_tp1, _ = model(
        SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], _is_training=True), [], None
    )

    target_model_out_tp1, _ = target_model(
        SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], _is_training=True), [], None
    )

    alpha = torch.exp(model.log_alpha)

    # Discrete case.
    if model.discrete:
        # Get all action probs directly from pi and form their logp.
        action_dist_inputs_t, _ = model.get_action_model_outputs(
            model_out_t)
        log_pis_t = F.log_softmax(action_dist_inputs_t, dim=-1)
        policy_t = torch.exp(log_pis_t)
        action_dist_inputs_tp1, _ = model.get_action_model_outputs(
            model_out_tp1)
        log_pis_tp1 = F.log_softmax(action_dist_inputs_tp1, -1)
        policy_tp1 = torch.exp(log_pis_tp1)

        # For critic loss. Q-values for all the critics
        q_ts, _ = model.get_q_values(model_out_t, num_critics=-1)
        one_hot = F.one_hot(
            train_batch[SampleBatch.ACTIONS].long(), num_classes=q_ts[0].size()[-1]
        )
        q_ts_selected = [torch.sum(q_t * one_hot, dim=-1) for q_t in q_ts]
        # For clipping the critic update
        if policy.config['value_function_clipping']:
            q_ts_target, _ = target_model.get_q_values(
                target_model_out_tp1, num_critics=-1)
            # one_hot = F.one_hot(
            #     train_batch[SampleBatch.ACTIONS].long(), num_classes=q_ts[0].size()[-1]
            # )

            q_ts_selected_target = [torch.sum(q_t * one_hot, dim=-1)
                                    for q_t in q_ts_target]

        # For target value update. Discrete case: "Best" means weighted by the policy (prob) outputs.
        q_tp1s, _ = target_model.get_q_values(
            target_model_out_tp1, num_critics=num_critics)
        # , axis=0)  # min, mean or median
        q_tp1 = target_predictor(q_tp1s) - alpha * log_pis_tp1
        # q_tp1 = res if type(res) is torch.Tensor else res[0]

        # q_tp1 -= alpha * log_pis_tp1

        q_tp1_best = torch.sum(torch.mul(policy_tp1, q_tp1), dim=-1)
        # q_tp1_best_masked = (1.0 - train_batch[SampleBatch.DONES].float()) * q_tp1_best

        # For actor loss.
        q_t_reparam_policy = q_predictor(q_ts)  # , axis=0)   # was mean
        # q_t_reparam_policy = q_t_reparam_policy if type(
        #     q_t_reparam_policy) is torch.Tensor else q_t_reparam_policy[0]
    # Continuous actions case.
    else:
        # Sample single actions from distribution.
        action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)
        action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
        action_dist_t = action_dist_class(action_dist_inputs_t, model)
        policy_t = (
            action_dist_t.sample()
            if not deterministic
            else action_dist_t.deterministic_sample()
        )
        log_pis_t = torch.unsqueeze(action_dist_t.logp(policy_t), -1)
        action_dist_inputs_tp1, _ = model.get_action_model_outputs(model_out_tp1)
        action_dist_tp1 = action_dist_class(action_dist_inputs_tp1, model)
        policy_tp1 = (
            action_dist_tp1.sample()
            if not deterministic
            else action_dist_tp1.deterministic_sample()
        )
        log_pis_tp1 = torch.unsqueeze(action_dist_tp1.logp(policy_tp1), -1)

        # For critic Loss. Q-values from all critics for the actually selected actions.
        q_ts, _ = model.get_q_values(
            model_out_t, train_batch[SampleBatch.ACTIONS], num_critics=-1)
        q_ts_selected = [torch.squeeze(q_t, dim=-1) for q_t in q_ts]
        # For clipping the critic update
        if policy.config['value_function_clipping']:
            q_ts_target, _ = target_model.get_q_values(
                target_model_out_tp1, policy_tp1, num_critics=-1)
            q_ts_selected_target = [torch.squeeze(q_t, dim=-1) for q_t in q_ts_target]

        # For target value update. Target Q network evaluation for sampled target critics
        q_tp1s, _ = target_model.get_q_values(
            target_model_out_tp1, policy_tp1, num_critics=num_critics)
        q_tp1 = target_predictor(q_tp1s) - alpha * log_pis_tp1  # , axis=0)
        # q_tp1 = res if type(res) is torch.Tensor else res[0]
        # q_tp1 -= alpha * log_pis_tp1

        q_tp1_best = torch.squeeze(input=q_tp1, dim=-1)

        # For actor loss. Q-values from all critics for current policy in given current state. For re-parametrization
        q_t_reparam_policy, _ = model.get_q_values(
            model_out_t, policy_t, num_critics=-1)
        q_t_reparam_policy = q_predictor(q_t_reparam_policy)  # , axis=0)
        # q_t_reparam_policy = res if type(res) is torch.Tensor else res[0]

    # compute RHS of bellman equation
    q_tp1_best_masked = (1.0 - train_batch[SampleBatch.DONES].float()) * q_tp1_best
    y_target = (
        train_batch[SampleBatch.REWARDS]
        + (policy.config["gamma"] ** policy.config["n_step"]) * q_tp1_best_masked
    ).detach()

    # Critic loss
    # Compute the mean TD-error for the ensemble (potentially clipped).
    td_error = 0
    critic_loss = []
    for idx in range(len(q_ts_selected)):
        cur_error = torch.abs(q_ts_selected[idx] - y_target)
        td_error += cur_error
        loss = torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(cur_error))
        if policy.config['value_function_clipping']:
            clip_err = torch.abs(
                q_ts_selected_target[idx] - y_target +
                torch.clamp(
                    input=q_ts_selected[idx] - q_ts_selected_target[idx],
                    min=-policy.config['value_function_clip_value'],
                    max=policy.config['value_function_clip_value']
                )
            )
            clip_loss = torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(clip_err))
            loss = torch.max(loss, clip_loss)
        critic_loss.append(
            loss
        )
    td_error /= len(q_ts_selected)

    # Alpha loss.
    # Note: In the papers, alpha is used directly, here we take the log.
    # Discrete case: Multiply the action probs as weights with the original
    # loss terms (no expectations needed).
    weighted_alpha = - alpha * (log_pis_t.detach() + model.target_entropy)
    if model.discrete:
        # Sum up weighted terms and mean over all batch items.
        alpha_loss = torch.sum(weighted_alpha, dim=-1)
    alpha_loss = torch.mean(weighted_alpha)
    # Actor loss
    if model.discrete:
        # # weighted_log_alpha_loss = policy_t.detach() * (
        # #     -model.log_alpha * (log_pis_t + model.target_entropy).detach()
        # # )
        # weighted_log_alpha_loss = -policy_t.detach() * (
        #     alpha * (log_pis_t.detach() + model.target_entropy)
        # )
        # # Sum up weighted terms and mean over all batch items.
        # alpha_loss = torch.mean(torch.sum(weighted_log_alpha_loss, dim=-1))
        # Actor loss.
        actor_loss = torch.mean(
            torch.sum(
                torch.mul(
                    # NOTE: No stop_grad around policy output here
                    # (compare with q_t_det_policy for continuous case).
                    policy_t,
                    alpha.detach() * log_pis_t - q_t_reparam_policy.detach(),
                ),
                dim=-1,
            )
        )
    else:
        # alpha_loss = -torch.mean(
        #     alpha * (log_pis_t.detach() + model.target_entropy)
        # )
        # Note: Do not detach q_t_det_policy here b/c is depends partly
        # on the policy vars (policy sample pushed through Q-net).
        # However, we must make sure `actor_loss` is not used to update
        # the Q-net(s)' variables.
        actor_loss = torch.mean(alpha.detach() * log_pis_t - q_t_reparam_policy)
    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["q_t"] = torch.stack(q_ts)
    model.tower_stats["policy_t"] = policy_t
    model.tower_stats["log_pis_t"] = log_pis_t
    model.tower_stats["actor_loss"] = actor_loss
    model.tower_stats["critic_loss"] = critic_loss
    model.tower_stats["alpha_loss"] = alpha_loss

    # TD-error tensor in final stats
    # will be concatenated and retrieved for each individual batch item.
    model.tower_stats["td_error"] = td_error

    # Return all loss terms corresponding to our optimizers.
    return tuple([actor_loss] + critic_loss + [alpha_loss])


# def stats(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
#     """Stats function for REDQ. Returns a dict with important loss stats.

#     Args:
#         policy: The Policy to generate stats for.
#         train_batch: The SampleBatch (already) used for training.

#     Returns:
#         Dict[str, TensorType]: The stats dict.
#     """
#     q_t = torch.stack(policy.get_tower_stats("q_t"))

#     return {
#         "actor_loss": torch.mean(torch.stack(policy.get_tower_stats("actor_loss"))),
#         "critic_loss": torch.mean(
#             torch.stack(tree.flatten(policy.get_tower_stats("critic_loss")))
#         ),
#         "alpha_loss": torch.mean(torch.stack(policy.get_tower_stats("alpha_loss"))),
#         "alpha_value": torch.exp(policy.model.log_alpha),
#         "log_alpha_value": policy.model.log_alpha,
#         "target_entropy": policy.model.target_entropy,
#         "policy_t": torch.mean(torch.stack(policy.get_tower_stats("policy_t"))),
#         "mean_q": torch.mean(q_t),
#         "max_q": torch.max(q_t),
#         "min_q": torch.min(q_t),
#     }


def optimizer_fn(policy: Policy, config: AlgorithmConfigDict) -> Tuple[LocalOptimizer]:
    """Creates all necessary optimizers for REDQ learning.

    The ensemble_size + 2 optimizers returned here correspond to the
    number of loss terms returned by the loss function.

    Args:
        policy: The policy object to be trained.
        config: The Algorithm's config dict.

    Returns:
        Tuple[LocalOptimizer]: The local optimizers to use for policy training.
    """
    policy.actor_optim = torch.optim.Adam(
        params=policy.model.policy_variables(),
        lr=config["optimization"]["actor_learning_rate"],
        eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
    )

    # don't quite understand why separate here
    ensemble_size = config['ensemble_size']
    critic_split = len(policy.model.q_variables()) // ensemble_size
    policy.critic_optims = []
    for idx in range(ensemble_size):
        policy.critic_optims.append(
            torch.optim.Adam(
                params=policy.model.q_variables(
                )[idx * critic_split:(idx + 1) * critic_split],
                lr=config["optimization"]["critic_learning_rate"],
                eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
            )
        )

    policy.alpha_optim = torch.optim.Adam(
        params=[policy.model.log_alpha],
        lr=config["optimization"]["entropy_learning_rate"],
        eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
    )
    return tuple([policy.actor_optim] + policy.critic_optims + [policy.alpha_optim])


# Build a child class of `TorchPolicy`, given the custom functions defined
# above.
REDQTorchPolicy = build_policy_class(
    name="REDQTorchPolicy",
    framework="torch",
    loss_fn=actor_critic_loss,
    get_default_config=lambda: redq.DEFAULT_CONFIG,
    stats_fn=stats,
    postprocess_fn=postprocess_trajectory,
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=optimizer_fn,
    validate_spaces=validate_spaces,
    before_loss_init=setup_late_mixins,
    make_model_and_action_dist=build_redq_model_and_action_dist,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    mixins=[TargetNetworkMixin, ComputeTDErrorMixin],
    action_distribution_fn=action_distribution_fn,
)
