"""V035 直立 velocity 任务（v033 E1 奖励档，18 全直驱关节）。

与 v034 的差异：
  * 机器人资产换成 v035（无闭链、无被动关节，站高 0.229577 m）；
  * 所有 ``knee_up`` 正则/字典改为 ``knee``，并补上 knee 的 4340 力矩上限；
  * 观测/动作维度 18（v034 观测 22，含 4 个被动关节）；
  * 删除 v033 关节表生成的 knee_up 遥测列，按 v035 关节表重建。
"""

from mjlab.asset_zoo.robots.pikachu.pikachu_v035_constants import (
  ACTUATOR_GROUPS,
  JOINT_NAMES,
  POLICY_ACTION_JOINT_NAMES,
  POLICY_OBSERVATION_JOINT_NAMES,
  STAND_HEIGHT,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pikachu_foot_telemetry import add_pikachu_foot_telemetry_metrics
from mjlab.tasks.pikachu_metrics import add_pikachu_telemetry_metrics
from mjlab.tasks.velocity.config.pikachu_v034_biped_e1.env_cfgs import (
  v034_biped_e1_env_cfg,
)

from . import mdp
from .robot import biped_e1_robot_cfg


def _drop_v034_metrics(cfg: ManagerBasedRlEnvCfg) -> None:
  """删掉 v033/v034 版本用 ``knee_up``/``knee_mid`` 注册的遥测列。"""
  for name in tuple(cfg.metrics):
    if "knee_up" in name or "knee_mid" in name:
      cfg.metrics.pop(name)


def v035_biped_e1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = v034_biped_e1_env_cfg(play=play)
  cfg.scene.entities["robot"] = biped_e1_robot_cfg()
  cfg.scene.num_envs = 1 if play else 1024
  cfg.rewards["base_height"].params["target_height"] = STAND_HEIGHT
  cfg.rewards["directional_progress"] = RewardTermCfg(
    func=mdp.directional_progress,
    weight=2.0,
  )

  # --- 手臂：v035 的 arm_roll 轴是 (-1,0,0)，v033/v034 是 (+1,0,0) --------
  # 复用 v034 的绝对角度约定时符号整体反了：旧默认位姿 L-53°/R+53° 在 v035 上
  # 把上臂抬到水平面上方 53°（渲染出来是"双手举起"），而 arm_position_window
  # 的 (45,53) 窗口也正好奖励这个举手姿态，所以六次训练都稳定停在 ±50°。
  # 默认位姿已在 pikachu_v035_constants.BIPED_KEYFRAME 改为 L+53°/R-53°
  # （手臂下垂 53°），这里把左右 arm_roll 的窗口镜像过来保持一致。
  # ARM_JOINTS 顺序 = 右臂 4 个 + 左臂 4 个，所以 roll 在 1 和 5。
  # 注意：elbow/arm_yaw/arm_pitch 的窗口不动——实测偏差 < 8°，罚项 < 0.5。
  bounds = list(cfg.rewards["arm_position_window"].params["bounds_degrees"])
  for index in (1, 5):
    lower, upper = bounds[index]
    bounds[index] = (-upper, -lower)
  cfg.rewards["arm_position_window"].params["bounds_degrees"] = tuple(bounds)

  # --- 动作：18 维、顺序固定为右腿/左腿/右臂/左臂 ------------------------
  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.actuator_names = POLICY_ACTION_JOINT_NAMES
  action.preserve_order = True
  action.scale = {
    r".*_hip_pitch_joint": 0.50,
    r".*_knee_joint": 0.40,
    r".*_hip_roll_joint": 0.20,
    r".*_hip_yaw_joint": 0.20,
    r".*_ankle_joint": 0.35,
    r".*_arm_pitch_joint": 0.35,
    r".*_arm_roll_joint": 0.30,
    r".*_arm_yaw_joint": 0.20,
    r".*_elbow_joint": 0.30,
  }

  # --- 观测：18 维，与动作同序 -------------------------------------------
  policy_joints = SceneEntityCfg(
    "robot", joint_names=POLICY_OBSERVATION_JOINT_NAMES, preserve_order=True
  )
  for group in cfg.observations.values():
    for name in ("joint_pos", "joint_vel"):
      group.terms[name].params["asset_cfg"] = policy_joints
  driven_joints = SceneEntityCfg("robot", joint_names=POLICY_OBSERVATION_JOINT_NAMES)

  # --- 奖励里所有 knee_up 正则换成 knee ---------------------------------
  pose = cfg.rewards["pose"]
  pose.params["asset_cfg"] = driven_joints
  for key in ("std_walking", "std_running"):
    std = pose.params[key]
    pose.params[key] = {
      (r".*knee.*" if pattern == r".*knee_up.*" else pattern): value
      for pattern, value in std.items()
    }
  standing = cfg.rewards["standing_posture_l2"]
  standing.params["asset_cfg"] = driven_joints
  standing.params["std"] = {
    (r".*knee.*" if pattern == r".*knee_up.*" else pattern): value
    for pattern, value in standing.params["std"].items()
  }
  cfg.rewards["standing_joint_velocity"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=JOINT_NAMES
  )

  # --- 扭矩软限位：knee 是 40 N·m 的 4340，其余沿用 4310 / 3507 ----------
  cfg.rewards["torque_soft_limit"].params = {
    "effort_limits": {
      r".*_knee_joint": 40.0,
      r".*(hip_pitch|hip_roll|hip_yaw|ankle|arm_pitch|arm_roll)_joint": 12.5,
      r".*(arm_yaw|elbow)_joint": 3.0,
    },
    "soft_ratio": 0.80,
    "asset_cfg": driven_joints,
  }

  # --- 遥测：删掉 knee_up 列后按 v035 关节表重建 --------------------------
  _drop_v034_metrics(cfg)
  add_pikachu_telemetry_metrics(
    cfg, joint_names=JOINT_NAMES, actuator_groups=ACTUATOR_GROUPS
  )
  if play:
    add_pikachu_foot_telemetry_metrics(
      cfg, joint_names=JOINT_NAMES, entity_name="robot"
    )
  return cfg
