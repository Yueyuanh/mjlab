"""Independent V034 flat-ground biped and crawl baselines."""

from dataclasses import fields

from mjlab.asset_zoo.robots.pikachu.pikachu_v034_constants import (
  FOOT_GEOMS,
  JOINT_NAMES,
  STAND_HEIGHT,
  get_biped_robot_cfg,
)
from mjlab.envs import mdp as common
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from . import crawl_mdp
from .command import CrawlVelocityCommandCfg
from .robot import SUPPORT_GEOMS, SUPPORT_SITES, get_crawl_robot_cfg


def _base(play=False, crawl=False):
  cfg = make_velocity_env_cfg()
  cfg.scene.entities = {
    "robot": get_crawl_robot_cfg() if crawl else get_biped_robot_cfg()
  }
  cfg.scene.num_envs = 1 if play else 256
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.curriculum = {}
  cfg.terminations.pop("out_of_terrain_bounds")
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != "terrain_scan")
  for group in cfg.observations.values():
    group.terms.pop("height_scan")
  foot_scan = cfg.scene.sensors[0]
  foot_scan.frame = tuple(
    ObjRef(type="site", name=f"{s}_foot", entity="robot") for s in ("left", "right")
  )
  foot_scan.pattern = RingPatternCfg.single_ring(radius=0.02, num_samples=4)
  contact = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="body",
      entity="robot",
      pattern=r"(left|right)_(ankle|elbow)_link"
      if crawl
      else r"(left|right)_ankle_link",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors += (contact,)
  cfg.sim.njmax = 300
  cfg.sim.nconmax = None
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.mujoco.iterations = 20
  cfg.sim.mujoco.timestep = 0.0025
  cfg.decimation = 8  # 50 Hz policy, 400 Hz physics for the closed-chain knees.
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.2
  cfg.actions["joint_pos"].scale = {
    r".*_hip_pitch_joint": 0.35,
    r".*_hip_roll_joint": 0.12,
    r".*_hip_yaw_joint": 0.12,
    r".*_knee_up_joint": 0.30,
    r".*_ankle_joint": 0.25,
    r".*_arm_pitch_joint": 0.25,
    r".*_arm_roll_joint": 0.20,
    r".*_arm_yaw_joint": 0.15,
    r".*_elbow_joint": 0.20,
  }
  twist = cfg.commands["twist"]
  twist.heading_command = False
  twist.ranges.heading = None
  for name in (
    "rel_heading_envs",
    "rel_world_envs",
    "rel_forward_envs",
    "rel_backward_envs",
    "rel_lateral_envs",
  ):
    setattr(twist, name, 0.0)
  twist.rel_standing_envs = 0.2
  twist.ranges.lin_vel_x = (-0.15, 0.25) if not crawl else (-0.08, 0.15)
  twist.ranges.lin_vel_y = (-0.08, 0.08) if not crawl else (-0.04, 0.04)
  twist.ranges.ang_vel_z = (-0.2, 0.2) if not crawl else (-0.15, 0.15)
  cfg.events.pop("push_robot")
  cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.0)
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    SUPPORT_GEOMS if crawl else FOOT_GEOMS
  )
  cfg.events["foot_friction"].params["ranges"] = (0.6, 1.0)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  cfg.events["base_com"].params["ranges"] = {i: (-0.003, 0.003) for i in range(3)}
  cfg.observations["actor"].terms["base_lin_vel"].noise.n_min = -0.05
  cfg.observations["actor"].terms["base_lin_vel"].noise.n_max = 0.05
  if play:
    cfg.observations["actor"].enable_corruption = False
  return cfg


def biped_env_cfg(play=False):
  cfg = _base(play=play)
  feet = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))
  driven = SceneEntityCfg("robot", joint_names=JOINT_NAMES)
  for name in ("upright", "body_ang_vel"):
    cfg.rewards[name].params["asset_cfg"].body_names = ("base_link",)
  pose = cfg.rewards["pose"]
  pose.params["asset_cfg"] = driven
  pose.params["std_standing"] = {".*": 0.15}
  pose.params["std_walking"] = {".*": 0.45}
  pose.params["std_running"] = {".*": 0.50}
  pose.weight = 0.5
  cfg.rewards["track_linear_velocity"].params["std"] = 0.12
  cfg.rewards["track_angular_velocity"].params["std"] = 0.25
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["dof_pos_limits"].params["asset_cfg"] = driven
  for name in ("foot_clearance", "foot_slip"):
    cfg.rewards[name].params["asset_cfg"] = feet
  for name in ("foot_clearance", "foot_swing_height"):
    cfg.rewards[name].params["target_height"] = 0.025
  cfg.rewards["air_time"].weight = 0.5
  cfg.rewards["air_time"].params["command_threshold"] = 0.05
  cfg.rewards["base_height"] = RewardTermCfg(
    func=mdp.base_height_penalty,
    weight=-0.5,
    params={"target_height": STAND_HEIGHT, "std": 0.04},
  )
  cfg.rewards["base_vertical_velocity"] = RewardTermCfg(
    func=mdp.base_vertical_velocity_penalty, weight=-0.5
  )
  cfg.rewards["alternating_air_time"] = RewardTermCfg(
    func=mdp.alternating_feet_air_time,
    weight=1.0,
    params={
      "sensor_name": "feet_ground_contact",
      "target_air_time": 0.20,
      "std": 0.10,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  )
  cfg.terminations["fell_over"].params["limit_angle"] = 0.8
  cfg.terminations["low_base"] = TerminationTermCfg(
    func=low_base, params={"minimum": 0.14}
  )
  mdp.add_velocity_tracking_metrics(cfg, command_name="twist")
  return cfg


def low_base(env, minimum):
  return (
    env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    < minimum
  )


def crawl_env_cfg(play=False):
  cfg = _base(play=play, crawl=True)
  cfg.scene.sensors = tuple(
    s for s in cfg.scene.sensors if s.name != "foot_height_scan"
  )
  for group in cfg.observations.values():
    group.terms.pop("foot_height", None)
    for name, func in (
      ("base_lin_vel", crawl_mdp.heading_lin_vel),
      ("base_ang_vel", crawl_mdp.heading_ang_vel),
    ):
      group.terms[name].func, group.terms[name].params = func, {}
    group.terms["crawl_phase"] = ObservationTermCfg(func=crawl_mdp.phase_observation)
  cfg.observations["critic"].terms["foot_contact_forces"] = ObservationTermCfg(
    func=crawl_mdp.contact_forces
  )
  twist = cfg.commands["twist"]
  cfg.commands["twist"] = CrawlVelocityCommandCfg(
    **{f.name: getattr(twist, f.name) for f in fields(twist)}
  )
  cfg.events["reset_base"] = EventTermCfg(func=crawl_mdp.reset_base, mode="reset")
  cfg.rewards = {
    "track_linear_velocity": RewardTermCfg(func=crawl_mdp.tracking, weight=3),
    "linear_velocity_huber": RewardTermCfg(func=crawl_mdp.tracking_huber, weight=-0.5),
    "track_angular_velocity": RewardTermCfg(
      func=crawl_mdp.tracking, weight=1, params={"angular": True}
    ),
    "crawl_orientation": RewardTermCfg(func=crawl_mdp.orientation_error, weight=-3),
    "crawl_height": RewardTermCfg(func=crawl_mdp.height_error, weight=-0.3),
    "crawl_pose": RewardTermCfg(func=crawl_mdp.posture, weight=-0.5),
    "support": RewardTermCfg(func=crawl_mdp.support_cost, weight=-0.5),
    "gait_contacts": RewardTermCfg(func=crawl_mdp.gait_contacts, weight=1.5),
    "swing_height": RewardTermCfg(
      func=crawl_mdp.swing_height,
      weight=-0.5,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", site_names=SUPPORT_SITES, preserve_order=True
        )
      },
    ),
    "support_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-10,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.0,
        "asset_cfg": SceneEntityCfg(
          "robot",
          site_names=("left_hand", "right_hand", "left_foot", "right_foot"),
          preserve_order=True,
        ),
      },
    ),
    "action_rate_l2": RewardTermCfg(func=common.action_rate_l2, weight=-0.05),
    "dof_pos_limits": RewardTermCfg(
      func=common.joint_pos_limits,
      weight=-2,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
    ),
    "vertical_velocity": RewardTermCfg(
      func=mdp.base_vertical_velocity_penalty, weight=-0.5
    ),
  }
  cfg.metrics["support_count"] = MetricsTermCfg(func=crawl_mdp.support_count)
  for axis, name in enumerate(("x", "y", "yaw")):
    cfg.metrics[f"vel/act_{name}"] = MetricsTermCfg(
      func=crawl_mdp.heading_velocity_metric, params={"axis": axis}
    )
    cfg.metrics[f"vel/err_{name}"] = MetricsTermCfg(
      func=crawl_mdp.heading_velocity_error_metric, params={"axis": axis}
    )
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=common.time_out, time_out=True),
    "fell_over": TerminationTermCfg(func=crawl_mdp.fell_over),
  }
  return cfg
