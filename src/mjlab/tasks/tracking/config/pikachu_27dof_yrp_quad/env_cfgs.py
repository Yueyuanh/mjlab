"""Pikachu **27-dof hip-YRP 四足** 前进 tracking 环境配置。

与 ``pikachu_27dof_quad``（PRY）同构，仅把模型换成 **YRP** hip 拆分序训练体
``pikachu_27dof_train_yrp.xml``（四足 keyframe/碰撞按名字解析，直接复用 PRY 四足
标定值；robot cfg 在此内联组装，避免跨 constants 依赖）。策略动作驱动 16 走步关节
（腿 10 + 前腿 arm_pitch/roll/elbow 6）跟踪四足参考；arm_yaw 固定 0、头/耳/尾固定。

参考 npz（``pikachu_quad.npz``）的 body 列在 **PRY 模型**上 FK 重放生成，靠
``motion_cmd.motion_body_column_names`` 运行时按名字对齐到 YRP 模型的
``pikachu_27dof_quad_walk_bodies()`` 序（见 ``commands.py``）。

> 运动学差异：YRP 与 PRY 的 hip 拆分轴序不同，非零关节角下 FK 末端位姿不等价，
> 即使对齐 PRY 参考仍会有 tracking 误差。为明示接受前提（npz 数据不动、只运行对齐）。
"""

from mjlab.asset_zoo.robots.pikachu_xml.pikachu_27dof_constants import (
  PIKACHU_27DOF_ARTICULATION,
)
from mjlab.asset_zoo.robots.pikachu_xml.pikachu_27dof_quad_constants import (
  QUAD_27DOF_ACTION_SCALE,
  QUAD_27DOF_COLLISION,
  QUAD_27DOF_KEYFRAME,
  QUAD_27DOF_MOTION_FILE,
  QUAD_27DOF_SUPPORT_GEOM_NAMES,
  QUAD_27DOF_WALK_JOINT_NAMES,
  pikachu_27dof_quad_walk_bodies,
)
from mjlab.asset_zoo.robots.pikachu_xml.pikachu_27dof_yrp_constants import (
  PIKACHU_27DOF_PRY_XML,
  get_body_column_names,
  get_yrp_spec,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.pikachu_metrics import add_pikachu_telemetry_metrics
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

# 躯干(基座)body 名。
ANCHOR_BODY_NAME = "base_link"

# 后端终止只保留四腿末端:后脚踝 + 前手掌(肘)。与参考 body 布局对齐。
EE_BODY_NAMES: tuple[str, ...] = (
  "left_ankle_link",
  "right_ankle_link",
  "left_elbow_link",
  "right_elbow_link",
)

# npz body 列序 = 原 PRY 训练模型 body 先序（生成时的布局）。运行时据此把参考列
# 按名字重排到 YRP 模型的 quad body 序（见 commands.py MotionLoader）。
MOTION_BODY_COLUMN_NAMES = get_body_column_names(PIKACHU_27DOF_PRY_XML)


def _quad_robot_cfg() -> EntityCfg:
  """27-dof hip-YRP 四足机器人配置（YRP spec + 四足标定 keyframe/碰撞）。"""
  return EntityCfg(
    init_state=QUAD_27DOF_KEYFRAME,
    collisions=(QUAD_27DOF_COLLISION,),
    spec_fn=get_yrp_spec,
    articulation=PIKACHU_27DOF_ARTICULATION,
  )


def pikachu_27dof_yrp_quad_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Pikachu 27-dof hip-YRP quadrilateral forward-walk tracking config."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": _quad_robot_cfg()}

  # 27dof box 碰撞体多,放宽接触容量。
  cfg.sim.njmax = 800
  cfg.sim.nconmax = 70
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64

  # 自碰撞传感器(tracking 奖励引用 name="self_collision")。
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

  # -- 动作层:裁剪驱动到 16 走步关节 ---------------------------------------
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = QUAD_27DOF_WALK_JOINT_NAMES
  joint_pos_action.scale = QUAD_27DOF_ACTION_SCALE

  # -- 参考运动命令:PRY 布局 -> 运行时按名字对齐到 YRP 四足 body/关节 -----------
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = str(QUAD_27DOF_MOTION_FILE)
  motion_cmd.anchor_body_name = ANCHOR_BODY_NAME
  motion_cmd.body_names = pikachu_27dof_quad_walk_bodies()
  motion_cmd.joint_names = QUAD_27DOF_WALK_JOINT_NAMES
  motion_cmd.motion_body_column_names = MOTION_BODY_COLUMN_NAMES
  # 跳过参考运动开头几帧(录制起步常夹杂沉地/未落定),避免 reset 一上来就被
  # contact 弹开。见 commands.py MotionCommandCfg.start_frame。
  motion_cmd.start_frame = 4

  # -- 事件 / 奖励 / 终止 ------------------------------------------------------
  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = QUAD_27DOF_SUPPORT_GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = (ANCHOR_BODY_NAME,)

  # joint limit 只限走步 16 关节(固定关节由 pose 锁住)。
  cfg.rewards["joint_limit"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=QUAD_27DOF_WALK_JOINT_NAMES
  )

  # 功率限制:惩罚 16 个走步关节的机械注入功率之和 Σ max(0, τ_actuator * qd) (W)。
  cfg.rewards["power"] = RewardTermCfg(
    func=mdp.electrical_power_cost,
    weight=-2e-3,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=QUAD_27DOF_WALK_JOINT_NAMES)
    },
  )

  cfg.terminations["ee_body_pos"].params["body_names"] = EE_BODY_NAMES
  cfg.viewer.body_name = ANCHOR_BODY_NAME

  add_pikachu_telemetry_metrics(cfg)
  # 27dof 无 foot site,移除依赖它的脚滑遥测。
  cfg.metrics.pop("foot_slip_velocity_m_s", None)
  cfg.metrics.pop("foot_slide_fraction", None)

  # -- 评估(play)模式:无限 episode、关随机化 ---------------------------------
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
