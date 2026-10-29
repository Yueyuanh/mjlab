"""V034: 18 driven joints, 4 passive four-bar joints, native CAD conventions.

Motor ratings come from the versioned robot_config.json. Rotor inertias and
10 Hz position gains are initial DM motor estimates, not hardware calibration.
"""

import json
import math

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

ASSET_DIR = MJLAB_SRC_PATH / "asset_zoo/robots/pikachu/xmls/pikachu_v034"
PIKACHU_V034_XML = ASSET_DIR / "mjcf/pikachu_v034_train.xml"
GEOMETRY = json.loads((ASSET_DIR / "geometry.json").read_text())
MOTOR_CONFIG = json.loads((ASSET_DIR / "config/robot_config.json").read_text())
STAND_HEIGHT = GEOMETRY["stand_height"]
FOOT_GEOMS = r"(left|right)_foot[1-5]_collision"
DRIVEN_KINDS = tuple(k for k, v in MOTOR_CONFIG["joints"].items() if v["motor_type"])
JOINT_NAMES = tuple(
  f"{side}_{kind}_joint" for side in ("left", "right") for kind in DRIVEN_KINDS
)
PASSIVE_JOINTS = tuple(
  f"{side}_{kind}_joint" for side in ("left", "right") for kind in ("knee", "knee_mid")
)


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(PIKACHU_V034_XML))


def _actuator(motor: str, armature: float) -> BuiltinPositionActuatorCfg:
  omega = 10 * 2 * math.pi
  return BuiltinPositionActuatorCfg(
    target_names_expr=tuple(
      f".*_{kind}_joint"
      for kind in DRIVEN_KINDS
      if MOTOR_CONFIG["joints"][kind]["motor_type"] == motor
    ),
    stiffness=armature * omega**2,
    damping=4 * armature * omega,
    effort_limit=MOTOR_CONFIG["motor_types"][motor]["torque_limit_nm"],
    armature=armature,
  )


ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    _actuator("4310", 2.206e-5 * 10**2),
    _actuator("3507", 6.69169e-6 * 21**2),
  ),
  soft_joint_pos_limit_factor=0.95,
)
BIPED_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, STAND_HEIGHT + 0.003),
  joint_pos={
    "left_arm_roll_joint": math.radians(-53),
    "right_arm_roll_joint": math.radians(53),
  },
  joint_vel={".*": 0.0},
)


def get_biped_robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_spec,
    init_state=BIPED_KEYFRAME,
    articulation=ARTICULATION,
    collisions=(
      CollisionCfg(
        geom_names_expr=(FOOT_GEOMS,),
        contype=1,
        conaffinity=1,
        condim=3,
        priority=1,
        friction=(0.8,),
      ),
    ),
  )
