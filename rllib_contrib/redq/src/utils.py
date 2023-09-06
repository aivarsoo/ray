import copy

import gym
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import AlgorithmConfigDict
from redq_torch_model import REDQTorchModel


def build_redq_model(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: AlgorithmConfigDict,
) -> ModelV2:
    """Constructs the necessary ModelV2 for the Policy and returns it.

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
    # Force-ignore any additionally provided hidden layer sizes.
    # Everything should be configured using SAC's `q_model_config` and
    # `policy_model_config` config settings.
    policy_model_config = copy.deepcopy(MODEL_DEFAULTS)
    policy_model_config.update(config["policy_model_config"])
    q_model_config = copy.deepcopy(MODEL_DEFAULTS)
    q_model_config.update(config["q_model_config"])
    if config["framework"] == "torch":
        default_model_cls = REDQTorchModel
        model_interface_cls = REDQTorchModel
    else:
        raise NotImplementedError

    model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=None,
        model_config=config["model"],
        framework=config["framework"],
        # model_interface=default_model_cls,
        default_model=default_model_cls,
        name="redq_model",
        policy_model_config=policy_model_config,
        q_model_config=q_model_config,
        ensemble_size=config["ensemble_size"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"],
    )

    assert isinstance(model, default_model_cls)

    # Create an exact copy of the model and store it in `policy.target_model`.
    # This will be used for tau-synched Q-target models that run behind the
    # actual Q-networks and are used for target q-value calculations in the
    # loss terms.
    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=None,
        model_config=config["model"],
        framework=config["framework"],
        # model_interface=model_interface_cls,
        default_model=default_model_cls,
        name="target_redq_model",
        policy_model_config=policy_model_config,
        q_model_config=q_model_config,
        ensemble_size=config["ensemble_size"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"],
    )

    assert isinstance(policy.target_model, default_model_cls)

    return model
