"""V035 四足 crawl 任务：直驱膝版本，沿用已选定的 v034 crawl-knee 奖励档。"""

from dataclasses import fields

from mjlab.asset_zoo.robots.pikachu.pikachu_v035_constants import (
  ACTUATOR_GROUPS,
  JOINT_NAMES,
  POLICY_ACTION_JOINT_NAMES,
  POLICY_OBSERVATION_JOINT_NAMES,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as common
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.pikachu_metrics import add_pikachu_telemetry_metrics
from mjlab.tasks.velocity import mdp as velocity
from mjlab.tasks.velocity.config.pikachu_v035 import crawl_mdp
from mjlab.tasks.velocity.config.pikachu_v035.command import CrawlVelocityCommandCfg
from mjlab.tasks.velocity.config.pikachu_v035.robot import (
  SUPPORT_GEOMS,
  SUPPORT_SITES,
  get_crawl_robot_cfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from . import mdp


def v035_crawl_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()
  cfg.scene.entities = {"robot": get_crawl_robot_cfg()}
  cfg.scene.num_envs = 1 if play else 1024
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.curriculum = {}
  cfg.terminations.pop("out_of_terrain_bounds")
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != "terrain_scan")
  for group in cfg.observations.values():
    group.terms.pop("height_scan", None)
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
  cfg.scene.sensors = tuple(
    s for s in cfg.scene.sensors if s.name != "foot_height_scan"
  )

  contact = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="body",
      entity="robot",
      pattern=r"(left|right)_(ankle|elbow)_link",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # 遥测脚速只应统计两只脚；crawl 的主接触传感器还包含两个肘部。
  feet_only = ContactSensorCfg(
    name="feet_only_ground_contact",
    primary=ContactMatch(
      mode="body", entity="robot", pattern=r"(left|right)_ankle_link"
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors += (contact, feet_only)
  cfg.sim.njmax = 300
  cfg.sim.nconmax = None
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.mujoco.iterations = 20
  cfg.sim.mujoco.timestep = 0.0025
  cfg.decimation = 8  # 50 Hz policy, 400 Hz physics
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.2
  cfg.actions["joint_pos"].scale = {
    r".*_hip_pitch_joint": 0.35,
    r".*_hip_roll_joint": 0.12,
    r".*_hip_yaw_joint": 0.12,
    r".*_knee_joint": 0.30,
    r".*_ankle_joint": 0.25,
    r".*_arm_pitch_joint": 0.25,
    r".*_arm_roll_joint": 0.20,
    r".*_arm_yaw_joint": 0.15,
    r".*_elbow_joint": 0.20,
  }

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
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
  twist.rel_standing_envs = 0.10
  twist.ranges.lin_vel_x = (-0.15, 0.20)
  twist.ranges.lin_vel_y = (-0.04, 0.04)
  twist.ranges.ang_vel_z = (-0.15, 0.15)
  if not play:
    twist.rel_forward_envs = 0.40
    twist.rel_backward_envs = 0.30
    twist.rel_lateral_envs = 0.10
  cfg.commands["twist"] = CrawlVelocityCommandCfg(
    **{f.name: getattr(twist, f.name) for f in fields(twist)}
  )

  cfg.events.pop("push_robot")
  cfg.events["reset_base"] = EventTermCfg(
    func=crawl_mdp.reset_base,
    mode="reset",
    params={"pose_range": {"z": (0.0, 0.0)}},
  )
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = SUPPORT_GEOMS
  cfg.events["foot_friction"].params["ranges"] = (0.6, 1.0)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  cfg.events["base_com"].params["ranges"] = {i: (-0.003, 0.003) for i in range(3)}
  cfg.observations["actor"].terms["base_lin_vel"].noise.n_min = -0.05
  cfg.observations["actor"].terms["base_lin_vel"].noise.n_max = 0.05
  if play:
    cfg.observations["actor"].enable_corruption = False

  # --- 奖励：v034 crawl-knee 档 ------------------------------------------
  cfg.rewards = {
    "track_linear_velocity": RewardTermCfg(func=crawl_mdp.tracking, weight=3),
    "linear_velocity_huber": RewardTermCfg(
      func=crawl_mdp.tracking_huber, weight=-1.0
    ),
    "track_angular_velocity": RewardTermCfg(
      func=crawl_mdp.tracking, weight=1, params={"angular": True}
    ),
    "crawl_orientation": RewardTermCfg(
      func=crawl_mdp.orientation_error, weight=-3
    ),
    "crawl_height": RewardTermCfg(func=crawl_mdp.height_error, weight=-0.3),
    "crawl_pose": RewardTermCfg(func=crawl_mdp.posture, weight=-0.5),
    "support": RewardTermCfg(func=crawl_mdp.support_cost, weight=-0.5),
    "gait_contacts": RewardTermCfg(func=crawl_mdp.gait_contacts, weight=0.0),
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
      func=velocity.feet_slip,
      weight=-20.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.0,
        "asset_cfg": SceneEntityCfg(
          "robot", site_names=SUPPORT_SITES, preserve_order=True
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
      func=velocity.base_vertical_velocity_penalty, weight=-0.5
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
    "elbow_window": RewardTermCfg(
      func=mdp.arm_posture_window,
      weight=-0.20,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=("right_elbow_joint", "left_elbow_joint"),
          preserve_order=True,
        ),
        "tolerance_degrees": 8.0,
      },
    ),
    "compact_endpoints": RewardTermCfg(
      func=mdp.endpoint_distance_window,
      weight=-0.40,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          site_names=("right_hand", "right_foot", "left_hand", "left_foot"),
          preserve_order=True,
        ),
        "minimum": 0.10,
        "maximum": 0.25,
        "std": 0.08,
      },
    ),
    "ankle_limit_margin": RewardTermCfg(
      func=mdp.ankle_limit_margin,
      weight=-1.0,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=("right_ankle_joint", "left_ankle_joint"),
          preserve_order=True,
        )
      },
    ),
    "gait_contacts_smooth_120": RewardTermCfg(
      func=mdp.smooth_gait_contacts,
      weight=1.8,
      params={"transition_seconds": 0.12},
    ),
    "forward_gait_mismatch_smooth_120": RewardTermCfg(
      func=mdp.directional_smooth_gait_mismatch,
      weight=-0.40,
      params={"transition_seconds": 0.12},
    ),
    "forward_velocity_shortfall": RewardTermCfg(
      func=mdp.forward_velocity_shortfall, weight=-0.60
    ),
    "directional_progress": RewardTermCfg(
      func=mdp.directional_progress, weight=3.0
    ),
    "action_acc_l2": RewardTermCfg(func=common.action_acc_l2, weight=0.0),
  }
  feet = SceneEntityCfg(
    "robot", body_names=("right_ankle_link", "left_ankle_link"), preserve_order=True
  )
  for name, weight, endpoint in (
    ("stance_sole_normal", -0.80, False),
    ("stance_sole_endpoints", -0.20, True),
    ("forward_sole_normal", -0.45, False),
    ("forward_sole_endpoints", -0.10, True),
  ):
    cfg.rewards[name] = RewardTermCfg(
      func=mdp.directional_sole_cost
      if name.startswith("forward")
      else mdp.sole_support_cost,
      weight=weight,
      params={"asset_cfg": feet, "endpoint": endpoint},
    )
  cfg.rewards["stance_sole_outlier"] = RewardTermCfg(
    func=mdp.sole_support_cost,
    weight=-0.50,
    params={
      "asset_cfg": feet,
      "tolerance_degrees": 10.0,
      "scale_degrees": 5.0,
    },
  )

  # --- 动作 / 观测顺序 ----------------------------------------------------
  action = cfg.actions["joint_pos"]
  action.actuator_names = POLICY_ACTION_JOINT_NAMES
  action.preserve_order = True
  policy_joints = SceneEntityCfg(
    "robot", joint_names=POLICY_OBSERVATION_JOINT_NAMES, preserve_order=True
  )
  for group in cfg.observations.values():
    for name in ("joint_pos", "joint_vel"):
      group.terms[name].params["asset_cfg"] = policy_joints

  # --- 终止与指标 ---------------------------------------------------------
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=common.time_out, time_out=True),
    "fell_over": TerminationTermCfg(func=crawl_mdp.fell_over),
  }
  cfg.metrics["support_count"] = MetricsTermCfg(func=crawl_mdp.support_count)
  for axis, name in enumerate(("x", "y", "yaw")):
    cfg.metrics[f"vel/act_{name}"] = MetricsTermCfg(
      func=crawl_mdp.heading_velocity_metric, params={"axis": axis}
    )
    cfg.metrics[f"vel/err_{name}"] = MetricsTermCfg(
      func=crawl_mdp.heading_velocity_error_metric, params={"axis": axis}
    )
  add_pikachu_telemetry_metrics(
    cfg,
    joint_names=JOINT_NAMES,
    actuator_groups=ACTUATOR_GROUPS,
    sensor_name="feet_only_ground_contact",
  )
  return cfg
