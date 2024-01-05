from __future__ import annotations

from typing import Dict
import safety_gymnasium
from safety_gymnasium.assets.geoms import Goal
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.bases.base_task import BaseTask
from safety_gymnasium.builder import Builder
from safety_gymnasium.utils.task_utils import get_task_class_name
        
class SimpleGoalLevel1(BaseTask):
    """
    Custom safety gym environment
    """
    def __init__(
        self, 
        config:Dict=dict(),
    ):
        super(SimpleGoalLevel1, self).__init__(config=config)
        # Increased difficulty and randomization
        # self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]

        # # Instantiate and register hazards
        # self._add_geoms(Hazards(
        #     num=1,
        #     size=0.7,
        #     locations=[(0,0)],
        #     is_lidar_observed=True,
        #     is_constrained=True,
        #     keepout=0.705))

        # # Instantiate and register the goal
        # self._add_geoms(Goal(
        #     keepout=0.305, 
        #     size=0.3, 
        #     locations=[(1.1, 1.1)],
        #     is_lidar_observed=True))
        
        # self.lidar_conf.max_dist = 3
        # self.lidar_conf.num_bins = 16
        
        # self.last_dist_goal = None
           
    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = 0.0
        dist_goal = self.dist_goal()
        reward += (self.last_dist_goal - dist_goal) * self.goal.reward_distance
        self.last_dist_goal = dist_goal

        if self.goal_achieved:
            reward += self.goal.reward_goal

        return reward
            
    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        self.build_goal_position()
        self.last_dist_goal = self.dist_goal()

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_goal() <= self.goal.size

tasks = {
    "SimpleGoalLevel1": SimpleGoalLevel1
}

class CustomBuilder(Builder):
   def _get_task(self):
        class_name = get_task_class_name(self.task_id)
        if class_name in tasks:
            task_class = tasks[class_name]
            task = task_class(config=self.config)
            task.build_observation_space()
        else:
            task = super()._get_task()    
        return task

from safety_gymnasium import __register_helper

if __name__ == "__main__":
    env_id = "SafetyPointGoalC-v0" 
    max_episode_steps = 1000
    config = {
        'agent_name': "Point",
        "lidar_conf.max_dist": 3,
        "lidar_conf.num_bins": 16,
        "placements_conf.extents": [-1.5, -1.5, 1.5, 1.5],
        # "last_dist_goal": None,
        "Hazards": dict(
            # name='hazards',
            num=1,
            size=0.7,
            locations=[(0,0)],
            is_lidar_observed=True,
            is_constrained=True,
            keepout=0.705),
        "Goal": dict(
            # name='goal',
            keepout=0.305,
            size=0.3,
            locations=[(1.1, 1.1)],
            is_lidar_observed=True)
        }
    __register_helper(
        env_id=env_id,
        entry_point='safety_gymnasium.builder:Builder',
        spec_kwargs={'config': config, 'task_id': env_id},
        max_episode_steps=max_episode_steps,
    )

    env = safety_gymnasium.make(env_id)
    s, i = env.reset()
    env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
    for k in range(1000):
        s, a, d, t, i = env.step(env.action_space.sample())
    print("dones")