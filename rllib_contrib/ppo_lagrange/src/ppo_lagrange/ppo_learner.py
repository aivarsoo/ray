from dataclasses import dataclass
from typing import Dict

from ray.rllib.algorithms.ppo.ppo_learner import PPOLearner
from ray.rllib.algorithms.ppo.ppo_learner import PPOLearnerHyperparameters
from ray.rllib.core.rl_module.rl_module import ModuleID
from ray.rllib.utils.annotations import override
from ray.rllib.utils.lambda_defaultdict import LambdaDefaultDict
from ray.rllib.utils.schedules.scheduler import Scheduler


LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY = "vf_loss_unclipped"
LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY = "vf_explained_var"
LEARNER_RESULTS_KL_KEY = "mean_kl_loss"
LEARNER_RESULTS_CURR_KL_COEFF_KEY = "curr_kl_coeff"
LEARNER_RESULTS_CURR_ENTROPY_COEFF_KEY = "curr_entropy_coeff"

LEARNER_RESULTS_CURR_LARGANGE_PENALTY_COEFF_KEY = "curr_lagrange_penalty_coeff"
MEAN_CONSTRAINT_VIOL = "mean_constraint_violation"
PENALTY = "penalty_coefficient"
P_PART = "P"
I_PART = "I"
D_PART = "D"


@dataclass
class PPOLagrangeLearnerHyperparameters(PPOLearnerHyperparameters):
    """Hyperparameters for the PPOLagrangeLearner sub-classes (framework specific).

    These should never be set directly by the user. Instead, use the PPOLagrangeConfig
    class to configure your algorithm.
    See `ray.rllib.algorithms.......::PPOLagrangeConfig::training()` for more details on the
    individual properties.
    """

    use_cost_critic: bool = None
    cvf_loss_coeff: float = None
    cvf_clip_param: float = None
    cost_limit: float = None
    penalty_coeff_config: Dict = None
    penalty_coefficient: float = None
    p_penalty_coefficient: float = None
    i_penalty_coefficient: float = None
    d_penalty_coefficient: float = None


class PPOLagrangeLearner(PPOLearner):
    @override(PPOLearner)
    def build(self) -> None:
        super().build()

        # Set up Lagrangian penalty coefficient variables (per module).
        # The penalty coefficient is update in
        # `self.additional_update_for_module()`.
        self.curr_lagrange_penalty_coeffs_per_module: Dict[ModuleID, Scheduler] = LambdaDefaultDict(
            lambda module_id: {
                PENALTY: self._get_tensor_variable(
                    self.hps.get_hps_for_module(module_id).penalty_coefficient
                ),
                P_PART: self._get_tensor_variable(
                    self.hps.get_hps_for_module(module_id).p_penalty_coefficient
                ),
                I_PART: self._get_tensor_variable(
                    self.hps.get_hps_for_module(module_id).i_penalty_coefficient
                ),
                D_PART: self._get_tensor_variable(
                    self.hps.get_hps_for_module(module_id).d_penalty_coefficient
                ),
            }
        )

    @override(PPOLearner)
    def remove_module(self, module_id: str):
        super().remove_module(module_id)
        self.curr_lagrange_penalty_coeffs_per_module.pop(module_id)
