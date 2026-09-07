"""Pikachu **27-dof sample-links** 双足 velocity 环境配置。

基于 ``make_velocity_env_cfg()`` 起步，机器人为 27-dof 拟真全身（腿 10 + 臂 8 +
头 3 + 耳 4 + 尾 2，见 ``asset_zoo/robots/pikachu_xml/pikachu_27dof_constants.py``）。
核心设定：

  - **策略动作只驱动腿部 10 关节**（走步），动作缩放与旧双足一致。
  - **上肢 17 关节作为随机摆动干扰**：两个零维副作用 ActionTerm
    ``RandomFlailAction``（见 ``envs/mdp/actions/random_flail.py``）——
    臂 8 关节（"手动随机摆动"）+ 耳 4/尾 2（"耳朵尾巴随机摆动"）在每个物理
    子步被写入**分段常数随机目标**（随机姿态 + 随机保持时长 → 突跳），制造
    外源动力学扰动。训练用课程把摆幅从 0 ramp 到上限，先学稳走再受扰。
  - **策略观测含全部 27 关节位置/速度**，能感知上肢摆动并主动平衡补偿。
  - 平地行走，只脚底碰撞。

脚底 site：29dof 训练体没有 ``left_foot``/``right_foot`` site，因此依赖足部
site 的 ``foot_clearance``/``foot_swing_height``/``foot_slip`` 奖励移除，改用
``alternating_feet_air_time`` + ``base_height`` 鼓励节律步态。
"""

from mjlab.asset_zoo.robots.pikachu_xml.pikachu_27dof_constants import (
  ARM_JOINT_NAMES,
  EXTREMITY_JOINT_NAMES,
  LEG_ACTION_SCALE,
  LEG_JOINT_NAMES,
  PIKACHU_27DOF_FOOT_GEOM_NAMES,
  get_pikachu_27dof_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, RandomFlailActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.pikachu_metrics import add_pikachu_telemetry_metrics
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

# 躯干(基座)body 名。
TORSO_BODY = "base_link"


def pikachu_27dof_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """创建 27-dof 双足平地 velocity 配置（腿上肢随机摆动干扰）。"""
  cfg = make_velocity_env_cfg()

  # -- 机器人实体与地形 ------------------------------------------------
  cfg.scene.entities = {"robot": get_pikachu_27dof_robot_cfg()}
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

  # 27dof box 碰撞体多,放宽接触容量(nefc overflow 风险)。
  cfg.sim.njmax = 800
  cfg.sim.nconmax = 70
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64

  # -- 足底接地传感器(双踝 body 子树)+ 自碰撞 -----------------------------
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
    primary=ContactMatch(mode="subtree", pattern=TORSO_BODY, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=TORSO_BODY, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg, self_collision_cfg)

  # -- 动作:策略只驱动腿 10 关节 + 上肢随机摆动干扰 -----------------------
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = LEG_JOINT_NAMES
  joint_pos_action.scale = LEG_ACTION_SCALE

  # "手动随机摆动":臂 8 关节; "耳朵尾巴随机摆动":耳 4 + 尾 2。
  cfg.actions["arm_flail"] = RandomFlailActionCfg(
    entity_name="robot",
    joint_names=ARM_JOINT_NAMES,
    flail_ratio=0.6,
    refresh_interval_range=(0.4, 1.0),
    curriculum_steps=(8000, 30000),
    use_curriculum=not play,
  )
  cfg.actions["extremity_flail"] = RandomFlailActionCfg(
    entity_name="robot",
    joint_names=EXTREMITY_JOINT_NAMES,
    flail_ratio=0.7,
    refresh_interval_range=(0.3, 0.8),
    curriculum_steps=(8000, 30000),
    use_curriculum=not play,
  )

  # -- 速度命令:双足全向,轻微前向偏好(同旧双足 flat) ----------------------
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_forward_envs = 0.25
  twist_cmd.rel_backward_envs = 0.25
  twist_cmd.rel_lateral_envs = 0.25
  twist_cmd.ranges.lin_vel_x = (-0.4, 0.5)
  twist_cmd.ranges.lin_vel_y = (-0.35, 0.35)
  twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {
      "step": 0,
      "lin_vel_x": (-0.4, 0.5),
      "lin_vel_y": (-0.35, 0.35),
      "ang_vel_z": (-0.3, 0.3),
    },
    {
      "step": 2000 * 24,
      "lin_vel_x": (-0.6, 0.8),
      "lin_vel_y": (-0.5, 0.5),
      "ang_vel_z": (-0.5, 0.5),
    },
  ]

  # -- 域随机化 --------------------------------------------------------
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    PIKACHU_27DOF_FOOT_GEOM_NAMES
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = (TORSO_BODY,)
  cfg.events["reset_robot_joints"].params["asset_cfg"].joint_names = (".*",)

  # -- 奖励 -----------------------------------------------------------
  cfg.rewards["upright"].params["asset_cfg"].body_names = (TORSO_BODY,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (TORSO_BODY,)

  # pose 正则:腿上肢宽松(上肢随 random_flail 摆动,不自锁);覆盖全部 27 关节。
  upper = (
    r".*_arm_pitch_joint",
    r".*_arm_roll_joint",
    r".*_arm_yaw_joint",
    r".*_elbow_joint",
    r"head_.*_joint",
    r".*_ear_.*_joint",
    r"tail_.*_joint",
  )
  leg = (
    r".*_hip_pitch_joint",
    r".*_hip_roll_joint",
    r".*_hip_yaw_joint",
    r".*_knee_joint",
    r".*_ankle_joint",
  )
  pose_std = {}
  for name in leg:
    pose_std[name] = {"stand": 0.05, "walk": 0.5, "run": 0.6}
  for name in upper:
    pose_std[name] = {"stand": 2.0, "walk": 2.0, "run": 2.0}
  for regime, key in (("std_standing", "stand"), ("std_walking", "walk"),
                      ("std_running", "run")):
    cfg.rewards["pose"].params[regime] = {
      name: vals[key] for name, vals in pose_std.items()
    }
  cfg.rewards["pose"].params["running_threshold"] = 2.0

  # joint limit 只限腿(上肢被 random_flail 有意推近限位)。
  cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=LEG_JOINT_NAMES
  )

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["action_rate_l2"].weight = -0.25
  cfg.rewards["air_time"] = RewardTermCfg(
    func=__import__(
      "mjlab.tasks.velocity.mdp", fromlist=["alternating_feet_air_time"]
    ).alternating_feet_air_time,  # noqa: E501
    weight=3.0,
    params={
      "sensor_name": feet_ground_cfg.name,
      "target_air_time": 0.22,
      "std": 0.08,
      "command_name": "twist",
      "command_threshold": 0.15,
    },
  )
  cfg.rewards["flight"] = RewardTermCfg(
    func=__import__(
      "mjlab.tasks.velocity.mdp", fromlist=["feet_flight_time"]
    ).feet_flight_time,  # noqa: E501
    weight=-2.0,
    params={
      "sensor_name": feet_ground_cfg.name,
      "grace_period": 0.04,
      "command_name": "twist",
      "command_threshold": 0.15,
    },
  )
  cfg.rewards["base_vertical_velocity"] = RewardTermCfg(
    func=__import__(
      "mjlab.tasks.velocity.mdp", fromlist=["base_vertical_velocity_penalty"]
    ).base_vertical_velocity_penalty,  # noqa: E501
    weight=-0.5,
  )
  cfg.rewards["base_height"] = RewardTermCfg(
    func=__import__(
      "mjlab.tasks.velocity.mdp", fromlist=["base_height_penalty"]
    ).base_height_penalty,  # noqa: E501
    weight=-0.2,
    params={"target_height": 0.17, "std": 0.015},
  )
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=__import__(
      "mjlab.tasks.velocity.mdp", fromlist=["self_collision_cost"]
    ).self_collision_cost,  # noqa: E501
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  # 27dof 无足部 site,移除依赖该 site 的奖励(初始起步不会 crash)。
  for name in ("foot_clearance", "foot_swing_height", "foot_slip", "soft_landing"):
    cfg.rewards.pop(name, None)

  # -- 查看器 / 遥测 ----------------------------------------------------
  cfg.viewer.body_name = TORSO_BODY
  add_pikachu_telemetry_metrics(cfg)
  # 27dof 无 foot site(left_foot/right_foot),移除依赖它的脚滑遥测。
  cfg.metrics.pop("foot_slip_velocity_m_s", None)
  cfg.metrics.pop("foot_slide_fraction", None)

  # -- 评估(play)模式:无限 episode、关随机化、干扰静止 -------------------
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