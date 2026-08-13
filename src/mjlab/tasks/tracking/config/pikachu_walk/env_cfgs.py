"""Pikachu forward-walking motion tracking environment configuration."""

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots.pikachu.pikachu_constants import (
  PIKACHU_FOOT_GEOM_NAMES,
  get_pikachu_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.pikachu_metrics import add_pikachu_telemetry_metrics
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

WALK_MOTION_FILE = str(
  MJLAB_SRC_PATH
  / "mocap"
  / "npz"
  / "pikachu_walk_npz"
  / "pikachu_walk_0.185_0.0_0.0.npz"
)

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
EE_BODY_NAMES: tuple[str, ...] = (
  "left_ankle_link",
  "right_ankle_link",
  "left_arm_yaw_link",
  "right_arm_yaw_link",
)

ACTION_SCALE: dict[str, float] = {
  r".*_hip_pitch_joint": 0.5,
  r".*_knee_joint": 0.5,
  r".*_hip_roll_joint": 0.35,
  r".*_hip_yaw_joint": 0.35,
  r".*_ankle_joint": 0.35,
  r".*_arm_pitch_joint": 0.35,
  r".*_arm_roll_joint": 0.35,
}


def pikachu_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Pikachu forward-walking tracking configuration."""
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

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = WALK_MOTION_FILE
  motion_cmd.anchor_body_name = ANCHOR_BODY_NAME
  motion_cmd.body_names = BODY_NAMES

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = PIKACHU_FOOT_GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = (ANCHOR_BODY_NAME,)

  cfg.terminations["ee_body_pos"].params["body_names"] = EE_BODY_NAMES
  cfg.viewer.body_name = ANCHOR_BODY_NAME

  add_pikachu_telemetry_metrics(cfg)

  # Match the transition task's contact capacity for the full Pikachu model.
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
