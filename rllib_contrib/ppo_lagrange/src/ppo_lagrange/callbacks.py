from typing import Dict
from typing import Tuple

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch


class ComputeEpisodeCostCallback(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == -1, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        # Create lists to store angles in
        episode.user_data["costs"] = []
        episode.hist_data["costs"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length >= 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        costs = []
        # TODO understand if that's a good way of doing it
        for agent in episode.get_agents():
            costs.append(episode._last_infos[agent]['cost'])
        episode.user_data["costs"].append(np.mean(costs))

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
        **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        # if worker.config.batch_mode == "truncate_episodes":
        # # Make sure this episode is really done.
        # assert episode.batch_builder.policy_collectors["default_policy"].batches[
        #     -1
        # ]["dones"][-1], (
        #     "ERROR: `on_episode_end()` should only be called "
        #     "after episode is done!"
        # )
        episode.custom_metrics["episode_cost"] = np.sum(episode.user_data["costs"])
        episode.hist_data["costs"] = episode.user_data["costs"]

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        result["callback_ok"] = True

        cost = result["custom_metrics"]["episode_cost"]
        result["custom_metrics"]["episode_cost_min"] = np.min(cost)
        result["custom_metrics"]["episode_cost_max"] = np.max(cost)
        result["custom_metrics"]["episode_cost_mean"] = np.mean(cost)
