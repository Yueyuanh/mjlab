"""Pikachu 状态转换（transition）tracking 环境配置。

本模块为 Pikachu 的"站↔趴"过渡段搭建 motion mimic（tracking）任务：
策略直接模仿一段参考动作，而不是靠奖励函数试错（见
``tasks/tracking/TRANSITION_ARCHITECTURE.md`` §3.3 与 §4）。

参考动作数据来自 ``mjlab.mocap`` 模块（PD 打点 + 关键帧插值 → CSV →
``csv_to_npz_pikachu.py`` 细化到 50fps）：
  - ``mocap/npz/Pikachu-Transition-Down.npz``：biped→quad（趴下）
  - ``mocap/npz/Pikachu-Transition-Up.npz``：quad→biped（站起）

两个方向动作模式完全不同，拆成两个独立模型（两个 checkpoint），
对应两个任务 ID：``Mjlab-Tracking-Flat-Pikachu-Down/Up``。

与 G1 tracking 配置（``config/g1/env_cfgs.py``）的主要差异（Pikachu 资产特性）：
  - 锚 / 躯干 body 是 ``base_link``（freejoint root body，XML 第一个 body）。
  - 手是 ``left/right_arm_yaw_link``（无执行器的 body，但有碰撞 geom）。
    四足撑地 / 起立时手必须被追踪，且需要 condim=3 + 摩擦（由
    ``FULL_COLLISION_WITHOUT_SELF`` 碰撞配置提供，即 ``get_pikachu_robot_cfg()``
    的默认值）。
  - 关节为 14 个（``hip_pitch/roll/yaw/knee/ankle/arm_pitch/arm_roll`` 左右各一）；
    动作缩放复用 velocity 任务验证过的 Pikachu 动作缩放。
"""

from typing import Literal

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots import get_pikachu_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

# 参考动作数据所在目录（mjlab.mocap 的输出，50fps）。
MOTION_DATA_DIR = MJLAB_SRC_PATH / "mocap" / "npz"

# biped→quad（趴下）参考动作。
TRANSITION_DOWN_MOTION_FILE = str(MOTION_DATA_DIR / "Pikachu-Transition-Down.npz")
# quad→biped（站起）参考动作。
TRANSITION_UP_MOTION_FILE = str(MOTION_DATA_DIR / "Pikachu-Transition-Up.npz")

# 追踪的 body 列表：锚（base_link）+ 双腿 hip/knee/ankle + 双手 arm_pitch/arm_yaw。
# arm_yaw_link 是手（四足撑地点），起立/趴下数据里手的位置与着地时机是核心，
# 必须追踪（对应 TRACKING_FLOW.md §7.3 的建议列表）。
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

# 末端（手/脚）body：用于 ee_body_pos 终止条件（末端脱离参考 z 时提前终止）。
EE_BODY_NAMES: tuple[str, ...] = (
  "left_ankle_link",
  "right_ankle_link",
  "left_arm_yaw_link",
  "right_arm_yaw_link",
)

# 脚底碰撞几何（Pikachu 为 G1 风格 foot1~7 胶囊，挂在 ankle body 子树内）。
FOOT_GEOM_NAMES = r"^(left|right)_foot[1-7]_collision$"

# 动作缩放：与 velocity 任务的 Pikachu 配置保持一致（hip_pitch/knee 幅度更大，
# 其余较小；见 tasks/velocity/config/pikachu/env_cfgs.py 的说明）。
ACTION_SCALE: dict[str, float] = {
  r".*_hip_pitch_joint": 0.5,
  r".*_knee_joint": 0.5,
  r".*_hip_roll_joint": 0.35,
  r".*_hip_yaw_joint": 0.35,
  r".*_ankle_joint": 0.35,
  r".*_arm_pitch_joint": 0.35,
  r".*_arm_roll_joint": 0.35,
}


def pikachu_transition_env_cfg(
  direction: Literal["down", "up"],
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Pikachu transition tracking configuration.

  Args:
    direction: 过渡方向。``"down"`` = biped→quad（趴下），``"up"`` = quad→biped
      （站起）。两者仅参考动作文件不同，其余配置完全一致。
    play: 是否启用 play（评估/播放）模式覆盖。

  Returns:
    一个 ManagerBasedRlEnvCfg 实例。
  """
  cfg = make_tracking_env_cfg()

  # 机器人实体：get_pikachu_robot_cfg() 默认 FULL_COLLISION_WITHOUT_SELF
  # （手脚都可与地面接触、手 condim=3 + 摩擦，满足四足撑地需求）。
  cfg.scene.entities = {"robot": get_pikachu_robot_cfg()}

  # 自碰撞传感器（整机 base_link 子树内）。相邻 link 接触已在 contact.xml
  # 排除，剩下的接触即异常碰撞，用于 self_collisions 惩罚。
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
  joint_pos_action.scale = ACTION_SCALE

  # Motion 命令：指向对应方向的参考动作，锚/body 列表按 Pikachu 资产设置。
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = ANCHOR_BODY_NAME
  motion_cmd.body_names = BODY_NAMES
  if direction == "down":
    motion_cmd.motion_file = TRANSITION_DOWN_MOTION_FILE
  else:
    motion_cmd.motion_file = TRANSITION_UP_MOTION_FILE

  # 事件：脚底摩擦随机化 + 质心偏移作用在 base_link 上。
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = (ANCHOR_BODY_NAME,)

  # 终止：末端（手脚）脱离参考 z 阈值提前终止。
  cfg.terminations["ee_body_pos"].params["body_names"] = EE_BODY_NAMES

  # 查看器跟踪躯干。
  cfg.viewer.body_name = ANCHOR_BODY_NAME

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


def pikachu_transition_down_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Pikachu transition-down (biped→quad) configuration."""
  return pikachu_transition_env_cfg("down", play=play)


def pikachu_transition_up_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Pikachu transition-up (quad→biped) configuration."""
  return pikachu_transition_env_cfg("up", play=play)
