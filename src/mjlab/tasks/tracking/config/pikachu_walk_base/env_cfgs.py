"""Pikachu walk-base 环境配置:纯腿 10 关节 + 摆臂课程干扰。

本任务右手臂消融实验的**真消融**:策略输入(观测)、输出(动作)、参考运动都只含
**10 个腿部关节**,手臂完全从网络剥离,由一个零维副作用 ActionTerm
``ArmSineAction`` 在每个物理子步施加正弦摆动(见 ``envs/mdp/actions/arm_sine.py``)。
训练中用课程把摆幅从 0 逐步 ramp 到上限,让下肢在学会稳定走路后才面对上肢移动的
动力学扰动;play 时由 ``ArmSwingViewer`` 滑条实时控制摆幅/频率。
"""

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots.pikachu.pikachu_constants import (
  PIKACHU_FOOT_GEOM_NAMES,
  get_pikachu_robot_cfg,
)
from mjlab.asset_zoo.robots.pikachu.pikachu_constants_base import LEG_JOINT_NAMES
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import ArmSineActionCfg, JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.pikachu_metrics import add_pikachu_telemetry_metrics
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

# 纯腿(前 10 dof)前进参考运动,与 walk-forward 模型训练数据对应。
WALK_BASE_MOTION_FILE = str(
  MJLAB_SRC_PATH
  / "mocap"
  / "npz"
  / "pikachu_walk_base_npz"
  / "pikachu_walk_0.185_base.npz"
)

ANCHOR_BODY_NAME = "base_link"

# 参考身体只保留腿:base + 每条腿的 hip_pitch/knee/ankle(与 walk 一致,去掉手臂)。
LEG_BODY_NAMES: tuple[str, ...] = (
  "base_link",
  "left_hip_pitch_link",
  "left_knee_link",
  "left_ankle_link",
  "right_hip_pitch_link",
  "right_knee_link",
  "right_ankle_link",
)

# 终止判定只保留双脚踝(手臂不再由策略控制,不参与跟踪终止)。
EE_BODY_NAMES: tuple[str, ...] = ("left_ankle_link", "right_ankle_link")

# 腿部动作缩放(与 walk 共享;arm 维已从策略移走,不再需要)。
LEG_ACTION_SCALE: dict[str, float] = {
  r".*_hip_pitch_joint": 0.5,
  r".*_knee_joint": 0.5,
  r".*_hip_roll_joint": 0.35,
  r".*_hip_yaw_joint": 0.35,
  r".*_ankle_joint": 0.35,
}


def pikachu_walk_base_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Pikachu walk-base (10-DOF legs + arm-swing) configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_pikachu_robot_cfg()}

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

  # -- 裁剪动作层到 10 腿 + 新增零维摆臂 term --------------------------------
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = LEG_JOINT_NAMES
  joint_pos_action.scale = LEG_ACTION_SCALE
  cfg.actions["arm_sine"] = ArmSineActionCfg(
    entity_name="robot",
    use_curriculum=not play,  # 训练开课程,play 由滑条直接控制幅值。
  )

  # -- 裁观测到腿 --------------------------------------------------------------
  leg_obs_cfg = SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)
  for group in cfg.observations.values():
    group.terms["joint_pos"].params["asset_cfg"] = leg_obs_cfg
    group.terms["joint_vel"].params["asset_cfg"] = leg_obs_cfg

  # -- 参考运动命令:10 dof 纯腿 -> 只用腿 body --------------------------------
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = WALK_BASE_MOTION_FILE
  motion_cmd.anchor_body_name = ANCHOR_BODY_NAME
  motion_cmd.body_names = LEG_BODY_NAMES
  motion_cmd.joint_names = LEG_JOINT_NAMES

  # -- 事件 / 奖励 / 终止 / 遥测 ----------------------------------------------
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = PIKACHU_FOOT_GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = (ANCHOR_BODY_NAME,)

  cfg.rewards["joint_limit"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=LEG_JOINT_NAMES
  )

  cfg.terminations["ee_body_pos"].params["body_names"] = EE_BODY_NAMES
  cfg.viewer.body_name = ANCHOR_BODY_NAME

  add_pikachu_telemetry_metrics(cfg)

  # 与 walk 相同的接触容量(全身 Pikachu 模型,含摆臂时的额外接触)。
  cfg.sim.nconmax = 70
  cfg.sim.njmax = 300

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
