"""Pikachu **27-dof 四足** 全向 velocity 环境配置。

基于 ``make_velocity_env_cfg()`` 起步,机器人为 27-dof 拟真全身(腿 10 + 臂 8 +
头 3 + 耳 4 + 尾 2)取**四足构型**:base 前俯 ≈107°,手臂当前腿(props掌着地)、
腿当后腿,四点撑地爬行。核心设定(与 14dof ``pikachu_quad`` 一脉相承):

  - **策略动作驱动 16 个走步关节**(腿 10 + 前腿 = 臂 pitch/roll/elbow 6)。
    ``arm_yaw`` 固定 0、头/耳/尾固定在四足关键帧姿态(pose 正则约束),不参与步态。
  - **四足前方系** 速度命令/观测/奖励/重置:直接复用 14dof 四足 ``quad_mdp.py``
    (``quad_front_lin_vel/ang_vel/command``、``quad_upright``、``quad_bad_orientation``、
    ``QuadUniformVelocityCommand``、``quad_reset_root_state_uniform``、``quad_track_*``),
    因读写 ``root_link_*`` 与关节数无关。
  - 命令走**世界系** ``vel_command_w``(base 前俯后 base-frame 前进语义失效),由
    ``QuadUniformVelocityCommandCfg`` 提供摇杆前方系写入 + 世界系 debug_vis。

起步姿态的落地配平(rot / arm/leg 关节 / pos.z / ``QUAD_27DOF_TILT_DEG`` /
``base_height`` target)沿用四足「门禁标定」哲学:先给初始值,viewer ``--agent zero``
实测后回填;27dof 头部偏重(≈11.96kg)尤其需要重标 settle 角。
"""

import dataclasses
import math

from mjlab.asset_zoo.robots.pikachu_xml.pikachu_27dof_quad_constants import (
  QUAD_27DOF_ACTION_SCALE,
  QUAD_27DOF_SUPPORT_GEOM_NAMES,
  QUAD_27DOF_TILT_DEG,
  QUAD_27DOF_TORSO_BODY,
  QUAD_27DOF_WALK_JOINT_NAMES,
  get_pikachu_27dof_quad_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.pikachu_metrics import add_pikachu_telemetry_metrics
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg

from ..pikachu_quad.quad_mdp import (
  QUAD_BODY_AXIS,
  quad_bad_orientation,
  quad_front_ang_vel,
  quad_front_command,
  quad_front_lin_vel,
  quad_reset_root_state_uniform,
  quad_track_angular_velocity,
  quad_track_linear_velocity,
  quad_upright,
)
from ..pikachu_quad.quad_mdp import (
  QuadUniformVelocityCommandCfg as QuadCmdCfg,
)

__all__ = ["pikachu_27dof_quad_flat_env_cfg"]

# 后脚 site(空气时间/脚打滑奖励载体;前手无 site,靠姿态+pose 支撑)。
_LEG_FOOT_SITES = ("left_foot", "right_foot")


def pikachu_27dof_quad_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """平坦地形 27-dof 四足全向 velocity 配置。"""
  cfg = make_velocity_env_cfg()

  # -- 机器人实体与地形 ------------------------------------------------
  cfg.scene.entities = {"robot": get_pikachu_27dof_quad_robot_cfg()}
  cfg.sim.njmax = 800  # 27dof box 碰撞体多,放宽接触容量。
  cfg.sim.nconmax = 70
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.curriculum.pop("terrain_levels", None)
  # flat 无地形:移除地形扫描。
  cfg.scene.sensors = tuple(
    s
    for s in (cfg.scene.sensors or ())
    if s.name not in ("terrain_scan", "foot_height_scan")
  )
  for grp in cfg.observations.values():
    grp.terms.pop("height_scan", None)
    grp.terms.pop("foot_height", None)

  # -- 接触传感器:后脚接地(后腿) + 自碰撞 ------------------------------
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree", pattern=r"^(left|right)_ankle_link$", entity="robot"
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=QUAD_27DOF_TORSO_BODY, entity="robot"),
    secondary=ContactMatch(
      mode="subtree", pattern=QUAD_27DOF_TORSO_BODY, entity="robot"
    ),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg, self_collision_cfg)

  # -- 动作:策略驱动 16 走步关节 + 头/耳/尾固定 --------------------------
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = QUAD_27DOF_WALK_JOINT_NAMES
  joint_pos_action.scale = QUAD_27DOF_ACTION_SCALE

  cfg.viewer.body_name = QUAD_27DOF_TORSO_BODY

  # -- 四足世界系命令类(摇杆前方系写 vel_command_w + 世界系 debug_vis) -----
  twist_cfg = cfg.commands["twist"]
  assert isinstance(twist_cfg, UniformVelocityCommandCfg)
  twist_cmd = QuadCmdCfg(
    **{f.name: getattr(twist_cfg, f.name) for f in dataclasses.fields(twist_cfg)}
  )
  cfg.commands["twist"] = twist_cmd
  twist_cmd.rel_standing_envs = 0.1
  twist_cmd.rel_world_envs = 1.0
  twist_cmd.rel_forward_envs = 0.0
  twist_cmd.rel_backward_envs = 0.0
  twist_cmd.rel_lateral_envs = 0.0
  twist_cmd.rel_heading_envs = 0.0  # 世界命令已含方向。
  twist_cmd.ranges.lin_vel_x = (-0.4, 0.5)
  twist_cmd.ranges.lin_vel_y = (-0.35, 0.35)
  twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)

  # -- 四足前方系观测(共享 body 系被 base 前俯污染) -----------------------
  for grp in ("actor", "critic"):
    cfg.observations[grp].terms["base_lin_vel"] = ObservationTermCfg(
      func=quad_front_lin_vel, noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5)
    )
    cfg.observations[grp].terms["base_ang_vel"] = ObservationTermCfg(
      func=quad_front_ang_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2)
    )
    cfg.observations[grp].terms["command"] = ObservationTermCfg(
      func=quad_front_command, params={"command_name": "twist"}
    )

  # -- 四足重置(绕世界竖直 yaw,保持前俯) --------------------------------
  cfg.events["reset_base"] = EventTermCfg(
    func=quad_reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (0.01, 0.05),
        "yaw": (-3.14, 3.14),
      },
      "velocity_range": {},
    },
  )

  # -- 四足前方系速度/朝向奖励 + 摔倒终止 --------------------------------
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=quad_track_linear_velocity,
    weight=2.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  )
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=quad_track_angular_velocity,
    weight=2.0,
    params={"command_name": "twist", "std": math.sqrt(0.5)},
  )
  cfg.rewards["upright"] = RewardTermCfg(
    func=quad_upright,
    weight=1.0,
    params={
      "std": math.sqrt(0.2),
      "tilt_deg": QUAD_27DOF_TILT_DEG,  # TODO(标定) 27dof 头部偏重需重标
      "body_axis": QUAD_BODY_AXIS,
      "asset_cfg": SceneEntityCfg("robot", body_names=(QUAD_27DOF_TORSO_BODY,)),
    },
  )
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=quad_bad_orientation,
    params={
      "limit_angle": math.radians(40.0),
      "tilt_deg": QUAD_27DOF_TILT_DEG,
      "body_axis": QUAD_BODY_AXIS,
      "asset_cfg": SceneEntityCfg("robot", body_names=(QUAD_27DOF_TORSO_BODY,)),
    },
  )

  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (QUAD_27DOF_TORSO_BODY,)
  cfg.rewards["angular_momentum"].weight = -0.02

  # -- pose 正则:腿 + 前腿 walking 档放宽;arm_yaw + 头/耳/尾固定小 std ----
  pose_std = {
    # 走步关节(腿 10 + 前腿 pitch/roll/elbow)迈步档。
    r".*_hip_pitch_joint": {"stand": 0.05, "walk": 0.4, "run": 0.5},
    r".*_knee_joint": {"stand": 0.05, "walk": 0.5, "run": 0.6},
    r".*_hip_roll_joint": {"stand": 0.05, "walk": 0.15, "run": 0.2},
    r".*_hip_yaw_joint": {"stand": 0.05, "walk": 0.15, "run": 0.2},
    r".*_ankle_joint": {"stand": 0.05, "walk": 0.35, "run": 0.35},
    r".*_arm_pitch_joint": {"stand": 0.05, "walk": 0.4, "run": 0.5},
    r".*_arm_roll_joint": {"stand": 0.05, "walk": 0.2, "run": 0.3},
    r".*_elbow_joint": {"stand": 0.05, "walk": 0.4, "run": 0.5},
    # 固定:arm_yaw + 头/耳/尾,小 std 一直锁在四足关键帧。
    r".*_arm_yaw_joint": {"stand": 0.05, "walk": 0.05, "run": 0.05},
    r"head_.*_joint": {"stand": 0.02, "walk": 0.02, "run": 0.02},
    r".*_ear_.*_joint": {"stand": 0.05, "walk": 0.05, "run": 0.05},
    r"tail_.*_joint": {"stand": 0.05, "walk": 0.05, "run": 0.05},
  }
  for regime, key in (
    ("std_standing", "stand"),
    ("std_walking", "walk"),
    ("std_running", "run"),
  ):
    cfg.rewards["pose"].params[regime] = {
      name: vals[key] for name, vals in pose_std.items()
    }
  cfg.rewards["pose"].params["running_threshold"] = 2.0

  # joint limit:只限走步 16 关节(固定关节由 pose 锁住)。
  cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=QUAD_27DOF_WALK_JOINT_NAMES
  )

  # -- 脚触地空气时间 / base 高度 / 其它沿用四足 ---------------------------
  cfg.rewards["air_time"] = RewardTermCfg(
    func=__import__(
      "mjlab.tasks.velocity.mdp", fromlist=["alternating_feet_air_time"]
    ).alternating_feet_air_time,  # noqa: E501
    weight=3.0,
    params={
      "sensor_name": "feet_ground_contact",
      "target_air_time": 0.22,
      "std": 0.08,
      "command_name": "twist",
      "command_threshold": 0.15,
    },
  )
  # base 高度:前俯起步的实测 root 离地。TODO(标定) 27dof 需实测回填,先沿用 14dof 值。
  cfg.rewards["base_height"] = RewardTermCfg(
    func=__import__(
      "mjlab.tasks.velocity.mdp", fromlist=["base_height_penalty"]
    ).base_height_penalty,  # noqa: E501
    weight=-0.2,
    params={"target_height": 0.177, "std": 0.02},
  )
  for name in ("foot_clearance", "foot_swing_height"):
    cfg.rewards.pop(name, None)
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = _LEG_FOOT_SITES

  # -- 事件:脚摩擦用手脚支撑 geom;重置全关节 ------------------------------
  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = QUAD_27DOF_SUPPORT_GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = (QUAD_27DOF_TORSO_BODY,)
  cfg.events["reset_robot_joints"].params["asset_cfg"].joint_names = (".*",)

  add_pikachu_telemetry_metrics(cfg)
  # 27dof 无 foot site,移除依赖它的脚滑遥测。
  cfg.metrics.pop("foot_slip_velocity_m_s", None)
  cfg.metrics.pop("foot_slide_fraction", None)

  # -- 评估(play)模式:无限 episode、关随机化 ------------------------------
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    twist_cmd.ranges.lin_vel_x = (-0.6, 0.8)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
