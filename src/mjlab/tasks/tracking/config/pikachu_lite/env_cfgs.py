"""Pikachu **lite 碰撞体** + RMG_AMP 25 条动作库的 tracking 任务配置。

与 ``pikachu_walk_base`` 同源(纯腿 10 关节 + 摆臂课程干扰),两处关键差异:

1. **碰撞体换成 lite**：``get_pikachu_robot_cfg`` / ``PIKACHU_FOOT_GEOM_NAMES``
   从 ``pikachu_constants_lite`` 导入。lite 的脚底是 ``ankle_link`` 自身网格
   (无 G1 风格 foot1~7 胶囊),17 个 ``<link>_collision`` geom。
2. **参考从单条变整库**：用 ``MultiMotionCommandCfg`` 吃
   ``mocap/npz/pikachu_rmgamp_npz`` 下全部 25 条 rmg 速度档 clip
   (vx∈[-0.37,0.37], vy∈[-0.2,0.2], wz∈[-0.278,0.278]),
   每个 env 独立持有 (clip_id, 帧号)。这就是下游"速度命令 -> 自动匹配"的
   **教师**：它见过整库、且观测里带完整参考。

任务 ID ``Mjlab-Tracking-Flat-Pikachu-Lite``,实验名 ``pikachu_lite_tracking``。
"""

from pathlib import Path

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots.pikachu.pikachu_constants_base import LEG_JOINT_NAMES
from mjlab.asset_zoo.robots.pikachu.pikachu_constants_lite import (
  PIKACHU_FOOT_GEOM_NAMES,
  get_pikachu_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import ArmSineActionCfg, JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.pikachu_metrics import add_pikachu_telemetry_metrics
from mjlab.tasks.tracking.mdp.multi_motion import MultiMotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

# RMG_AMP 转换产物：25 条 clip,每条 499 帧 @50fps。
RMGAMP_NPZ_DIR: Path = MJLAB_SRC_PATH / "mocap" / "npz" / "pikachu_rmgamp_npz"

ANCHOR_BODY_NAME = "base_link"

# 参考身体只保留腿(与 walk_base 一致:base + 每条腿 hip_pitch/knee/ankle)。
LEG_BODY_NAMES: tuple[str, ...] = (
  "base_link",
  "left_hip_pitch_link",
  "left_knee_link",
  "left_ankle_link",
  "right_hip_pitch_link",
  "right_knee_link",
  "right_ankle_link",
)

# 终止判定只保留双脚踝。
EE_BODY_NAMES: tuple[str, ...] = ("left_ankle_link", "right_ankle_link")

LEG_ACTION_SCALE: dict[str, float] = {
  r".*_hip_pitch_joint": 0.5,
  r".*_knee_joint": 0.5,
  r".*_hip_roll_joint": 0.35,
  r".*_hip_yaw_joint": 0.35,
  r".*_ankle_joint": 0.35,
}


def rmgamp_motion_files() -> tuple[str, ...]:
  """参考库的 25 条 npz(按文件名排序,顺序即 clip_id)。"""
  files = sorted(RMGAMP_NPZ_DIR.glob("*.npz"))
  if not files:
    raise FileNotFoundError(
      f"no reference npz under {RMGAMP_NPZ_DIR}; run "
      "`uv run python -m mjlab.scripts.csv_to_npz_rmgamp` first"
    )
  return tuple(str(p) for p in files)


def pikachu_lite_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Pikachu lite multi-clip tracking configuration."""
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

  # -- 裁剪动作层到 10 腿 + 零维摆臂 term -------------------------------------
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = LEG_JOINT_NAMES
  joint_pos_action.scale = LEG_ACTION_SCALE
  cfg.actions["arm_sine"] = ArmSineActionCfg(
    entity_name="robot",
    use_curriculum=not play,
  )

  # -- 裁观测到腿 --------------------------------------------------------------
  leg_obs_cfg = SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)
  for group in cfg.observations.values():
    group.terms["joint_pos"].params["asset_cfg"] = leg_obs_cfg
    group.terms["joint_vel"].params["asset_cfg"] = leg_obs_cfg

  # -- 参考动作命令：整库(25 条) ----------------------------------------------
  base_cmd = cfg.commands["motion"]
  assert hasattr(base_cmd, "pose_range")
  motion_files = rmgamp_motion_files()
  motion_cmd = MultiMotionCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=True,
    pose_range=dict(base_cmd.pose_range),
    velocity_range=dict(base_cmd.velocity_range),
    joint_position_range=base_cmd.joint_position_range,
    # 基类/训练脚本都要求 motion_file 是个存在的路径(``train.py`` 会拿它
    # 判空并 fallback 到 wandb registry)。多动作模式下真正加载的是
    # motion_files,这里指向第一条只是为了满足那条检查。
    motion_file=motion_files[0],
    motion_files=motion_files,
    anchor_body_name=ANCHOR_BODY_NAME,
    body_names=LEG_BODY_NAMES,
    joint_names=LEG_JOINT_NAMES,
  )
  cfg.commands["motion"] = motion_cmd

  # -- 事件 / 奖励 / 终止 / 遥测 ----------------------------------------------
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = PIKACHU_FOOT_GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = (ANCHOR_BODY_NAME,)

  cfg.rewards["joint_limit"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=LEG_JOINT_NAMES
  )

  cfg.terminations["ee_body_pos"].params["body_names"] = EE_BODY_NAMES
  cfg.viewer.body_name = ANCHOR_BODY_NAME

  add_pikachu_telemetry_metrics(cfg)

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
