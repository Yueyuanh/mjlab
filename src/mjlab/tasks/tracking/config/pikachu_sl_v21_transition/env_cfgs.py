"""Pikachu v2.1 sample-links 状态转换（transition）tracking 环境配置。

与旧 ``pikachu_transition``（capsule 碰撞体）"同构型"复用同一参考 npz
（``Pikachu-Transition-Down/Up.npz``）：机器人为 ``pikachu_sl_v21_train.xml``
（14 关节、17-body 顺序对齐 npz、手脚碰撞体），只替换机器人实体与脚底/手
碰撞正则。手为 ``arm_yaw_link`` 刚性占位体 box，四足撑地需 condim=3 + 摩擦
（由 ``get_pikachu_sl21_robot_cfg()`` 默认的 ``FULL_COLLISION_WITHOUT_SELF``
碰撞配置提供）。
"""

from typing import Literal

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots.pikachu.pikachu_constants_v21 import (
  PIKACHU_SL21_ACTION_SCALE,
  PIKACHU_SL21_FOOT_GEOM_NAMES,
  get_pikachu_sl21_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.pikachu_metrics import add_pikachu_telemetry_metrics
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

# 参考动作数据所在目录（mjlab.mocap 的输出，50fps）。沿用旧 transition 数据。
MOTION_DATA_DIR = MJLAB_SRC_PATH / "mocap" / "npz"

# biped→quad（趴下）参考动作。
TRANSITION_DOWN_MOTION_FILE = str(MOTION_DATA_DIR / "Pikachu-Transition-Down.npz")
# quad→biped（站起）参考动作。
TRANSITION_UP_MOTION_FILE = str(MOTION_DATA_DIR / "Pikachu-Transition-Up.npz")

# 追踪的 body 列表（与旧 transition 相同）：锚 + 双腿 + 双手（含 arm_yaw）。
ANCHOR_BODY_NAME = "base_link"
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

# 末端（手/脚）body：用于 ee_body_pos 终止条件。
EE_BODY_NAMES: tuple[str, ...] = (
  "left_ankle_link",
  "right_ankle_link",
  "left_arm_yaw_link",
  "right_arm_yaw_link",
)


def pikachu_sl_v21_transition_env_cfg(
  direction: Literal["down", "up"],
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create v2.1 sample-links Pikachu transition tracking configuration.

  Args:
    direction: 过渡方向。``"down"`` = biped→quad（趴下），``"up"`` = quad→biped
      （站起）。两者仅参考动作文件不同，其余配置完全一致。
    play: 是否启用 play（评估/播放）模式覆盖。

  Returns:
    一个 ManagerBasedRlEnvCfg 实例。
  """
  cfg = make_tracking_env_cfg()

  # 机器人实体：v2.1 sample-links（默认 FULL_COLLISION_WITHOUT_SELF，
  # 手脚都可与地面接触、手 condim=3 + 摩擦，满足四足撑地需求）。
  cfg.scene.entities = {"robot": get_pikachu_sl21_robot_cfg()}

  # 自碰撞传感器（整机 base_link 子树内）。
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

  # 动作缩放：复用 Pikachu velocity 验证过的动作幅度。
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PIKACHU_SL21_ACTION_SCALE

  # Motion 命令：指向对应方向的参考动作，锚/body 列表按 v2.1 资产设置。
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = ANCHOR_BODY_NAME
  motion_cmd.body_names = BODY_NAMES
  if direction == "down":
    motion_cmd.motion_file = TRANSITION_DOWN_MOTION_FILE
  else:
    motion_cmd.motion_file = TRANSITION_UP_MOTION_FILE

  # 事件：脚底摩擦随机化 + 质心偏移作用在 base_link 上。
  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = PIKACHU_SL21_FOOT_GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = (ANCHOR_BODY_NAME,)

  # 终止：末端（手脚）脱离参考 z 阈值提前终止。
  cfg.terminations["ee_body_pos"].params["body_names"] = EE_BODY_NAMES

  # 查看器跟踪躯干。
  cfg.viewer.body_name = ANCHOR_BODY_NAME

  # 与 velocity 任务共享同一套电机与机体状态 Metrics。
  add_pikachu_telemetry_metrics(cfg)

  # 过渡涉及手脚四点支撑，接触较多，放宽接触/约束数组上限。
  cfg.sim.nconmax = 70
  cfg.sim.njmax = 300

  # 评估（play）模式覆盖：无限回合、关闭随机化、从第 0 帧起播。
  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # 关闭 reset 状态随机化（RSI）。
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    # play 时固定从首帧开始，按参考帧逐帧播放。
    motion_cmd.sampling_mode = "start"

  return cfg


def pikachu_sl_v21_transition_down_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create v2.1 transition-down (biped→quad) configuration."""
  return pikachu_sl_v21_transition_env_cfg("down", play=play)


def pikachu_sl_v21_transition_up_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create v2.1 transition-up (quad→biped) configuration."""
  return pikachu_sl_v21_transition_env_cfg("up", play=play)
