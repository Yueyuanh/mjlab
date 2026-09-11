"""Pikachu **27-dof sample-links** 前进 tracking 环境配置。

基于 ``make_tracking_env_cfg()`` 起步，机器人为 27-dof 拟真全身（腿 10 + 臂 8 +
耳 4 + 尾 2）。核心设定与 velocity 27dof 一脉相承：

  - **策略动作只驱动腿部 10 关节**（跟踪参考走步）。
  - **上肢 17 关节作为随机摆动干扰**：两个零维副作用 ActionTerm
    ``RandomFlailAction``（臂 8 关节 + 耳 4/尾 2）写入分段常数随机目标，制造外源
    动力学扰动；训练用课程把摆幅 ramp 到上限，play 时干扰静止。
  - **策略观测覆盖全部 27 关节**（joint_pos_rel / joint_vel_rel 无裁剪），能感知
    上肢摆动并主动平衡补偿。
  - 参考运动为**纯腿前进走**（``pikachu_walk_0.185_base.npz``，腿 10 dof），上肢不参与
    跟踪；参考身体只保留 base + 双腿 hip_pitch/knee/ankle。

脚底：27dof box 机器人无 foot site，tracking 的 body 跟踪奖励/终止直接基于
``base_link`` 与双腿 link，不依赖足部 site。
"""

from mjlab import MJLAB_SRC_PATH
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
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.pikachu_metrics import add_pikachu_telemetry_metrics
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

# 躯干(基座)body 名。
ANCHOR_BODY_NAME = "base_link"

# 参考运动：纯腿 10 dof 前进走，重排到 27dof body 布局（见
# scripts/make_27dof_walk_reference.py），使 MotionLoader 能按 27dof 机器人 body
# id 正确切片。上肢由 random_flail 干扰，不参与跟踪。
WALK_BASE_MOTION_FILE = str(
  MJLAB_SRC_PATH
  / "mocap"
  / "npz"
  / "pikachu_walk_27dof_npz"
  / "pikachu_walk_0.185_27dof.npz"
)

# 参考身体只保留腿：base + 双腿 hip_pitch/knee/ankle（上肢随机摆动不参与跟踪）。
LEG_BODY_NAMES: tuple[str, ...] = (
  "base_link",
  "left_hip_pitch_link",
  "left_knee_link",
  "left_ankle_link",
  "right_hip_pitch_link",
  "right_knee_link",
  "right_ankle_link",
)

# 终止判定只保留双脚踝（上肢由 random_flail 独立干扰，不参与跟踪终止）。
EE_BODY_NAMES: tuple[str, ...] = ("left_ankle_link", "right_ankle_link")


def pikachu_27dof_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Pikachu 27-dof forward-walk tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_pikachu_27dof_robot_cfg()}

  # 27dof box 碰撞体多，放宽接触容量（velocity 27dof 同款）。
  cfg.sim.njmax = 800
  cfg.sim.nconmax = 70
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64

  # 自碰撞传感器（tracking 奖励引用 name="self_collision"）。
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=ANCHOR_BODY_NAME, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=ANCHOR_BODY_NAME, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  # -- 动作层：裁剪驱动到 10 腿 + 上肢随机摆动干扰 ------------------------------
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = LEG_JOINT_NAMES
  joint_pos_action.scale = LEG_ACTION_SCALE

  # "手动随机摆动"：臂 8 关节；"耳朵尾巴随机摆动"：耳 4 + 尾 2。
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

  # -- 参考运动命令：10 dof 纯腿 -> 只用腿 body --------------------------------
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = WALK_BASE_MOTION_FILE
  motion_cmd.anchor_body_name = ANCHOR_BODY_NAME
  motion_cmd.body_names = LEG_BODY_NAMES
  motion_cmd.joint_names = LEG_JOINT_NAMES

  # -- 事件 / 奖励 / 终止 ------------------------------------------------------
  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = PIKACHU_27DOF_FOOT_GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = (ANCHOR_BODY_NAME,)

  # joint limit 只限腿（上肢被 random_flail 有意推近限位）。
  cfg.rewards["joint_limit"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=LEG_JOINT_NAMES
  )

  cfg.terminations["ee_body_pos"].params["body_names"] = EE_BODY_NAMES
  cfg.viewer.body_name = ANCHOR_BODY_NAME

  add_pikachu_telemetry_metrics(cfg)
  # 27dof 无 foot site(left_foot/right_foot),移除依赖它的脚滑遥测。
  cfg.metrics.pop("foot_slip_velocity_m_s", None)
  cfg.metrics.pop("foot_slide_fraction", None)

  # -- 评估(play)模式：无限 episode、关随机化、干扰静止 -------------------------
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
