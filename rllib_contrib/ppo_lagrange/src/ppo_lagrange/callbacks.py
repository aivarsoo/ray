from typing import Dict
from typing import Tuple

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from collections import deque

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
        episode.user_data["costs"] = deque([],maxlen=1001)
        episode.hist_data["costs"] = deque([],maxlen=1001)

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
        costs = deque([], maxlen=1500)
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
        episode_cost = np.sum(episode.user_data["costs"])
        episode.custom_metrics["episode_cost"] = episode_cost
        # episode.hist_data["costs"] = episode.user_data["costs"]

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        result["callback_ok"] = True

        cost = result["custom_metrics"].get("episode_cost", 0)
        result["custom_metrics"]["episode_cost_min"] = np.min(cost)
        result["custom_metrics"]["episode_cost_max"] = np.max(cost)
        result["custom_metrics"]["episode_cost_mean"] = np.mean(cost)
        # result["episode_cost_min"] = np.min(cost)
        # result["episode_cost_max"] = np.max(cost)
        result['sampler_results']["episode_cost"] = np.mean(cost)


    # def post_process_metrics(adapt_iter, workers, metrics):
    #     # Obtain Current Dataset Metrics and filter out
    #     name = "_adapt_" + str(adapt_iter) if adapt_iter > 0 else ""

    #     # Only workers are collecting data
    #     res = collect_metrics(remote_workers=workers.remote_workers())

    #     metrics["episode_reward_max" + str(name)] = res["episode_reward_max"]
    #     metrics["episode_reward_mean" + str(name)] = res["episode_reward_mean"]
    #     metrics["episode_reward_min" + str(name)] = res["episode_reward_min"]

    #     return metrics