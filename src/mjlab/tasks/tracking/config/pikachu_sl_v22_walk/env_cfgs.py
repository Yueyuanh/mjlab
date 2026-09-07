"""Pikachu v2.2 sample-links 正向行走 tracking 环境配置。

与旧 ``pikachu_walk``（capsule 碰撞体）"同构型"复用同一参考 npz：
机器人为 ``pikachu_sl_v22_train.xml``（14 关节、17-body 顺序对齐 npz，
髋距收窄到 y=0.0515、腿更短(L71)，修复 v2.1 左右腿间距过大的笨重感），
无 head，关节轴向/限位服从旧约定。只替换机器人实体与脚底/手碰撞正则。
"""

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots.pikachu.pikachu_constants_v22 import (
  PIKACHU_SL22_ACTION_SCALE,
  PIKACHU_SL22_FOOT_GEOM_NAMES,
  get_pikachu_sl22_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.pikachu_metrics import add_pikachu_telemetry_metrics
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

# 与旧 walk 任务共用同一参考 npz（同构型迭代，npz 数据沿用）。
WALK_MOTION_FILE = str(
  MJLAB_SRC_PATH
  / "mocap"
  / "npz"
  / "pikachu_walk_stand_npz"
  / "Pikachu-rmg-stand-0.2.npz"
)

ANCHOR_BODY_NAME = "base_link"
# 追踪 body 列表与旧 walk 相同（锚 + 双腿 + 双手，含 arm_yaw 刚性占位体）。
BODY_NAMES: tuple[str, ...] = (
  "base_link",
  "left_hip_pitch_link",
  "left_knee_link",
  "left_ankle_link",
  "right_hip_pitch_link",
  "right_knee_link",
  "right_ankle_link",
  "left_arm_pitch_link",
  "left_arm_yaw_link",
  "right_arm_pitch_link",
  "right_arm_yaw_link",
)
EE_BODY_NAMES: tuple[str, ...] = (
  "left_ankle_link",
  "right_ankle_link",
  "left_arm_yaw_link",
  "right_arm_yaw_link",
)


def pikachu_sl_v22_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the v2.2 sample-links Pikachu forward-walking tracking config."""
  cfg = make_tracking_env_cfg()

  # 机器人实体：v2.2 sample-links（默认 FULL_COLLISION_WITHOUT_SELF）。
  cfg.scene.entities = {"robot": get_pikachu_sl22_robot_cfg()}

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

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PIKACHU_SL22_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = WALK_MOTION_FILE
  motion_cmd.anchor_body_name = ANCHOR_BODY_NAME
  motion_cmd.body_names = BODY_NAMES

  # v2.2 脚底 geom 正则（ankle_link box）。
  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = PIKACHU_SL22_FOOT_GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = (ANCHOR_BODY_NAME,)

  cfg.terminations["ee_body_pos"].params["body_names"] = EE_BODY_NAMES
  cfg.viewer.body_name = ANCHOR_BODY_NAME

  add_pikachu_telemetry_metrics(cfg)

  # 全 Pikachu 模型（手脚支撑）的接触/约束数组上限。
  # njmax=600：v2.2 用 box 碰撞体（脚/手 condim=3 + 多角点），实际 nefc 需
  #   ~337；300 会触发 "nefc overflow - please increase njmax"，约束行被丢弃。
  cfg.sim.nconmax = 70
  cfg.sim.njmax = 600

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
