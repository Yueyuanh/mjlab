"""Shared telemetry metrics for all Pikachu tasks."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

PIKACHU_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_joint",
  "left_arm_pitch_joint",
  "left_arm_roll_joint",
  "right_arm_pitch_joint",
  "right_arm_roll_joint",
)


def add_pikachu_telemetry_metrics(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add physical joint and base telemetry shared by every Pikachu task.

  The live Viser values remain signed. Episode summaries use the absolute peak,
  which is more useful when comparing actuator requirements across runs.
  """
  cfg.metrics["base_height_m"] = MetricsTermCfg(func=envs_mdp.root_height)
  for joint_name in PIKACHU_JOINT_NAMES:
    asset_cfg = SceneEntityCfg("robot", joint_names=(joint_name,))
    cfg.metrics[f"joint_position_rad/{joint_name}"] = MetricsTermCfg(
      func=envs_mdp.joint_position,
      params={"asset_cfg": asset_cfg},
      reduce="max_abs",
    )
    cfg.metrics[f"joint_velocity_rad_s/{joint_name}"] = MetricsTermCfg(
      func=envs_mdp.joint_velocity,
      params={"asset_cfg": asset_cfg},
      reduce="max_abs",
    )
    cfg.metrics[f"joint_torque_nm/{joint_name}"] = MetricsTermCfg(
      func=envs_mdp.joint_torque,
      params={"asset_cfg": asset_cfg},
      reduce="max_abs",
    )
