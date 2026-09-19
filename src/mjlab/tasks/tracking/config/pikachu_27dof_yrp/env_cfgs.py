"""Pikachu **27-dof hip-YRP** 前进 tracking 环境配置。

与 ``pikachu_27dof``（PRY）同构，仅把机器人换成 **YRP** hip 拆分序训练体
``pikachu_27dof_train_yrp.xml``。策略动作驱动腿 10 关节跟踪纯腿参考走步，上肢 8 +
耳 4 + 尾 2 作为随机摆动干扰（``RandomFlailAction``），策略观测覆盖全部 27 关节。

**数据对齐（关键）**：参考 npz（``pikachu_walk_0.185_27dof.npz``）的 body 列是在**原
PRY 模型**上生成的，其 ``body_*_w`` 列序 = PRY 模型 body 先序。YRP 模型同名 body 在
body 先序里的位置不同，直接按 id 切片会错列。本配置通过
``motion_cmd.motion_body_column_names`` 把这份 PRY body 名序交给 ``MotionLoader``，
在运行时按名字把参考列重排到当前 ``LEG_BODY_NAMES`` 序 —— 同名 body 是跨模型唯一
锚，与 hip 拆分轴序无关（见 ``commands.py``）。

> 注意：YRP 与 PRY 的 hip 拆分轴序不同，非零关节角下 FK 末端位姿数学上不等价，
> 因此即便对齐，PRY 参考位姿喂给 YRP 机器人会存在 tracking 误差。此为明示接受前提
> （npz 数据不动、只运行对齐）。

参考运动：纯腿 10 dof 前进走（重排到 27dof body 布局，见
``scripts/make_27dof_walk_reference.py``）。参考身体只保留 base + 双腿
hip_pitch/knee/ankle，上肢随机摆动不参与跟踪。
"""

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots.pikachu_xml.pikachu_27dof_constants import (
  ARM_JOINT_NAMES,
  EXTREMITY_JOINT_NAMES,
  LEG_ACTION_SCALE,
  LEG_JOINT_NAMES,
  PIKACHU_27DOF_FOOT_GEOM_NAMES,
)
from mjlab.asset_zoo.robots.pikachu_xml.pikachu_27dof_yrp_constants import (
  PIKACHU_27DOF_PRY_XML,
  get_body_column_names,
  get_pikachu_27dof_yrp_robot_cfg,
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

# 参考运动：纯腿 10 dof 前进走，重排到 27dof body 布局（由 PRY 模型生成）。
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

# npz body 列序 = 原 PRY 训练模型 body 先序（生成时的布局）。运行时据此把参考列
# 按名字重排到 YRP 模型的 LEG_BODY_NAMES 序（见 commands.py MotionLoader）。
MOTION_BODY_COLUMN_NAMES = get_body_column_names(PIKACHU_27DOF_PRY_XML)


def pikachu_27dof_yrp_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Pikachu 27-dof hip-YRP forward-walk tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_pikachu_27dof_yrp_robot_cfg()}

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

  # -- 参考运动命令：10 dof 纯腿 -> 只用腿 body；运行时按名字对齐 PRY 布局 --------
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = WALK_BASE_MOTION_FILE
  motion_cmd.anchor_body_name = ANCHOR_BODY_NAME
  motion_cmd.body_names = LEG_BODY_NAMES
  motion_cmd.joint_names = LEG_JOINT_NAMES
  motion_cmd.motion_body_column_names = MOTION_BODY_COLUMN_NAMES

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
