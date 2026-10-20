"""Independent V033 crawl task with four contact channels and horizontal commands."""

from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as common
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp as velocity
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg

from . import mdp
from .command import CrawlVelocityCommandCfg
from .robot import SUPPORT_GEOMS, SUPPORT_SITES, robot_cfg


def crawl_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()
  cfg.scene.entities = {"robot": robot_cfg()}
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.num_envs = 1024
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(
        mode="body", pattern=r"(right|left)_(ankle|elbow)_link", entity="robot"
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      track_air_time=True,
    ),
  )
  for group in cfg.observations.values():
    group.terms.pop("height_scan", None)
    group.terms.pop("foot_height", None)
    group.terms["base_lin_vel"].func = mdp.heading_lin_vel
    group.terms["base_lin_vel"].params = {}
    group.terms["base_ang_vel"].func = mdp.heading_ang_vel
    group.terms["base_ang_vel"].params = {}
  cfg.observations["critic"].terms["foot_contact_forces"] = ObservationTermCfg(
    func=mdp.contact_forces
  )
  cfg.observations["actor"].terms["base_lin_vel"].noise = UniformNoiseCfg(
    n_min=-0.05, n_max=0.05
  )
  cfg.observations["actor"].terms["base_ang_vel"].noise = UniformNoiseCfg(
    n_min=-0.10, n_max=0.10
  )
  for group in cfg.observations.values():
    group.terms["crawl_phase"] = ObservationTermCfg(func=mdp.phase_observation)
  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  # Keep the full actuator interface; posture constraints belong in rewards.
  action.actuator_names = (".*",)
  action.scale = {
    r".*_hip_pitch_joint": 0.35,
    r".*_hip_roll_joint": 0.15,
    r".*_hip_yaw_joint": 0.15,
    r".*_knee_up_joint": 0.25,
    r".*_ankle_joint": 0.25,
    r".*_arm_pitch_joint": 0.25,
    r".*_arm_roll_joint": 0.25,
    r".*_arm_yaw_joint": 0.20,
    r".*_elbow_joint": 0.25,
  }
  base_command = cfg.commands["twist"]
  assert isinstance(base_command, UniformVelocityCommandCfg)
  command = CrawlVelocityCommandCfg(
    **{field.name: getattr(base_command, field.name) for field in fields(base_command)}
  )
  cfg.commands["twist"] = command
  command.heading_command = False
  command.ranges.heading = None
  command.rel_heading_envs = 0
  command.rel_forward_envs = 0
  command.rel_backward_envs = 0
  command.rel_lateral_envs = 0
  command.rel_standing_envs = 0.10
  command.debug_vis = True
  command.viz.z_offset = 0.15
  command.ranges.lin_vel_x = (0.08, 0.20)
  command.ranges.lin_vel_y = (0.0, 0.0)
  command.ranges.ang_vel_z = (0.0, 0.0)
  cfg.curriculum = {
    "command_vel": CurriculumTermCfg(
      func=velocity.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {
            "step": 0,
            "lin_vel_x": (0.08, 0.20),
            "lin_vel_y": (0.0, 0.0),
            "ang_vel_z": (0.0, 0.0),
          },
          {
            "step": 1000 * 24,
            "lin_vel_x": (-0.10, 0.25),
            "lin_vel_y": (-0.08, 0.08),
            "ang_vel_z": (-0.20, 0.20),
          },
        ],
      },
    )
  }
  cfg.events["reset_base"] = EventTermCfg(func=mdp.reset_base, mode="reset")
  cfg.events.pop("push_robot", None)
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = SUPPORT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.005, 0.005),
    1: (-0.005, 0.005),
    2: (-0.005, 0.005),
  }
  cfg.rewards = {
    "track_linear_velocity": RewardTermCfg(func=mdp.tracking, weight=3),
    "linear_velocity_huber": RewardTermCfg(func=mdp.tracking_huber, weight=-0.75),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.tracking, weight=1, params={"angular": True}
    ),
    "crawl_orientation": RewardTermCfg(func=mdp.orientation_error, weight=-3),
    "crawl_height": RewardTermCfg(func=mdp.height_error, weight=-0.3),
    "crawl_pose": RewardTermCfg(func=mdp.posture, weight=-0.5),
    "support": RewardTermCfg(func=mdp.support_cost, weight=-0.5),
    "gait_contacts": RewardTermCfg(func=mdp.gait_contacts, weight=1.5),
    "swing_height": RewardTermCfg(
      func=mdp.swing_height,
      weight=-0.5,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", site_names=SUPPORT_SITES, preserve_order=True
        ),
      },
    ),
    "swing_air_time": RewardTermCfg(
      func=velocity.feet_air_time,
      weight=0.3,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.03,
        "threshold_max": 0.3,
        "command_name": "twist",
        "command_threshold": 0.04,
      },
    ),
    "support_slip": RewardTermCfg(
      func=velocity.feet_slip,
      weight=-18.0,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg(
          "robot", site_names=SUPPORT_SITES, preserve_order=True
        ),
        "command_name": "twist",
        "command_threshold": 0.0,
      },
    ),
    "action_rate_l2": RewardTermCfg(func=common.action_rate_l2, weight=-0.05),
    "dof_pos_limits": RewardTermCfg(func=common.joint_pos_limits, weight=-2),
    "vertical_velocity": RewardTermCfg(
      func=velocity.base_vertical_velocity_penalty, weight=-0.5
    ),
  }
  cfg.metrics = {"support_count": MetricsTermCfg(func=mdp.support_count)}
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=common.time_out, time_out=True),
    "fell_over": TerminationTermCfg(func=mdp.fell_over),
  }
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.2
  cfg.sim.njmax = 300
  cfg.sim.nconmax = None
  cfg.sim.contact_sensor_maxmatch = 64
  if play:
    cfg.scene.num_envs = 1
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    command.ranges.lin_vel_x = (-0.10, 0.25)
    command.ranges.lin_vel_y = (-0.08, 0.08)
    command.ranges.ang_vel_z = (-0.20, 0.20)
  return cfg
