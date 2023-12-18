
import numpy as np


import numpy as np
from typing import Tuple, Dict, List 
from safety_gym.envs.engine import Engine
from gym.utils import seeding
from gymnasium import Env


pointgoal = {'num_steps': 1000, 
             'action_noise': 0.0, 
             'placements_extents': [-1.5, -1.5, 1.5, 1.5], 
             'placements_margin': 0.0, 
             'floor_display_mode': False, 
             'robot_placements': None, 
             'robot_locations': [], 
             'robot_keepout': 0.4, 
             'robot_base': 'xmls/point.xml', 
             'robot_rot': None, 
             'randomize_layout': True, 
             'build_resample': True, 
             'continue_goal': True, 
             'terminate_resample_failure': True, 
             'observation_flatten': True, 
             'observe_sensors': True, 
             'observe_goal_dist': False, 
             'observe_goal_comp': False, 
             'observe_goal_lidar': True, 
             'observe_box_comp': False, 
             'observe_box_lidar': True, 
             'observe_circle': False, 
             'observe_remaining': False, 
             'observe_walls': False, 
             'observe_hazards': True, 
             'observe_vases': True, 
             'observe_pillars': False, 
             'observe_buttons': False, 
             'observe_gremlins': False, 
             'observe_vision': False, 
             'observe_qpos': False, 
             'observe_qvel': False, 
             'observe_ctrl': False, 
             'observe_freejoint': False, 
             'observe_com': False, 
             'render_labels': False, 
             'render_lidar_markers': True, 
             'render_lidar_radius': 0.15, 
             'render_lidar_size': 0.025, 
             'render_lidar_offset_init': 0.5, 
             'render_lidar_offset_delta': 0.06, 
             'vision_size': (60, 40), 
             'vision_render': True, 
             'vision_render_size': (300, 200), 
             'lidar_num_bins': 16, 
             'lidar_max_dist': 3, 
             'lidar_exp_gain': 1.0, 
             'lidar_type': 'pseudo', 
             'lidar_alias': True, 
             'compass_shape': 2, 
             'task': 'goal', 
             'goal_placements': None, 
             'goal_locations': [], 
             'goal_keepout': 0.305, 
             'goal_size': 0.3, 
             'box_placements': None, 
             'box_locations': [], 
             'box_keepout': 0.2, 
             'box_size': 0.2, 
             'box_density': 0.001,
             'box_null_dist': 2,
             'reward_distance': 1.0, 
             'reward_goal': 1.0, 
             'reward_box_dist': 1.0, 
             'reward_box_goal': 1.0, 
             'reward_orientation': False, 
             'reward_orientation_scale': 0.002, 
             'reward_orientation_body': 'robot',
             'reward_exception': -10.0,
             'reward_x': 1.0,
             'reward_z': 1.0, 
             'reward_circle': 0.1, 
             'reward_clip': 10,
             'buttons_num': 0,
             'buttons_placements': None,
             'buttons_locations': [],'buttons_keepout': 0.3, 'buttons_size': 0.1, 'buttons_cost': 1.0, 'buttons_resampling_delay': 10, 
             'circle_radius': 1.5, 
             'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'], 
             'sensors_hinge_joints': True, 'sensors_ball_joints': True, 'sensors_angle_components': True, 
             'walls_num': 0, 
             'walls_placements': None, 
             'walls_locations': [], 'walls_keepout': 0.0, 'walls_size': 0.5, 
             'constrain_hazards': True, 'constrain_vases': False, 'constrain_pillars': False, 'constrain_buttons': False, 'constrain_gremlins': False, 'constrain_indicator': True, 
             'hazards_num': 8, 
             'hazards_placements': None, 
             'hazards_locations': [], 'hazards_keepout': 0.18, 'hazards_size': 0.2, 'hazards_cost': 1.0, 
             'vases_num': 1, 
             'vases_placements': None, 
             'vases_locations': [], 'vases_keepout': 0.15, 'vases_size': 0.1, 'vases_density': 0.001, 'vases_sink': 4e-05, 'vases_contact_cost': 1.0, 'vases_displace_cost': 0.0, 'vases_displace_threshold': 0.001, 'vases_velocity_cost': 1.0, 'vases_velocity_threshold': 0.0001, 
             'pillars_num': 0, 
             'pillars_placements': None, 
             'pillars_locations': [], 'pillars_keepout': 0.3, 'pillars_size': 0.2, 'pillars_height': 0.5, 'pillars_cost': 1.0, 
             'gremlins_num': 0, 
             'gremlins_placements': None, 
             'gremlins_locations': [], 'gremlins_keepout': 0.5, 'gremlins_travel': 0.3, 'gremlins_size': 0.1, 'gremlins_density': 0.001, 'gremlins_contact_cost': 1.0, 'gremlins_dist_threshold': 0.2, 'gremlins_dist_cost': 1.0, 
             'frameskip_binom_n': 10, 
             'frameskip_binom_p': 1.0, 
             '_seed': None
             }

class CustomEngine(Engine):
    """
    Custom safety gym environment
    """
    def __init__(
        self, 
        env_name:str="",
        engine_cfg:Dict=None,
    ):
        super(CustomEngine, self).__init__(engine_cfg)
        self.max_episode_steps = engine_cfg['num_steps']
        self.env_name = env_name

    def seed(self, seed:int=None) -> List[int]:
        super(CustomEngine, self).seed(seed)
        self.np_random, seed = seeding.np_random(self._seed)
        return [seed]

    def step(self, action:np.ndarray) -> Tuple[np.ndarray, int, bool, Dict]:
        obs, reward, done, info = super(CustomEngine, self).step(action)
        info['pos_com'] = self.world.robot_com() # saving position of the robot to plot        
        return obs, reward, done, info
    
    def reset(self) -> Tuple:
        obs = super(CustomEngine, self).reset()
        return obs 