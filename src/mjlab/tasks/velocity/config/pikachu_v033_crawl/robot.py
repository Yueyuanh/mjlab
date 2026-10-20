"""V033 crawl asset with a user-adjusted rest pose, isolated from the biped."""

import math

import mujoco

from mjlab.asset_zoo.robots.pikachu.pikachu_v033_constants import (
  PIKACHU_V033_ARTICULATION,
  get_spec,
)
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

PITCH = 1.3
HEIGHT = 0.2314027754812906
SUPPORT_SITES = ("right_foot", "left_foot", "right_hand", "left_hand")
SUPPORT_GEOMS = r"(left|right)_(foot[1-5]|elbow_link)_collision"
# Contact reference points on the original STL surface at the requested pose.
# These sites have no collision or mass; elbow mesh geoms provide contact.
HAND_SUPPORT_POS = {
  "right": (0.0, -0.076, -0.021),
  "left": (0.0, 0.076, 0.021),
}
# Legs retain the calibrated IK and four-bar closure error < 6e-9 m.
# Arm roll/yaw/elbow and mesh support were checked on 2026-10-20.
RIGHT_POSE = {
  "hip_pitch": 0.6268899279922713,
  "hip_roll": -0.10212739168737714,
  "hip_yaw": 0.047219709416852246,
  "knee_up": 0.18803520715096875,
  "knee_mid": -0.1883920901551333,
  "knee": 0.164184111166522,
  "ankle": -0.5113024764854394,
  "arm_pitch": 1.4195450459376606,
  "arm_roll": -math.pi / 2,
  "arm_yaw": math.pi / 2,
  "elbow": 0.0,
}
JOINT_POSE = {
  f"{side}_{kind}_joint": value
  * (1 if side == "right" or kind in ("knee_up", "knee_mid", "arm_yaw") else -1)
  for side in ("right", "left")
  for kind, value in RIGHT_POSE.items()
}
# Both yaw joint coordinates are +90 degrees; arm roll retains mirrored signs.


def crawl_spec() -> mujoco.MjSpec:
  spec = get_spec()
  for side, pos in HAND_SUPPORT_POS.items():
    body = spec.body(f"{side}_elbow_link")
    body.add_site(name=f"{side}_hand", pos=pos, size=[0.003])
  # Physical torso contact and termination prevent underground reward exploits.
  spec.body("base_link").add_geom(
    name="crawl_torso_collision",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=[0.025, 0, 0.07],
    size=[0.045, 0.065, 0.09],
    mass=0,
    group=3,
  )
  return spec


def robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=crawl_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0, 0, HEIGHT + 0.003),
      rot=(math.cos(PITCH / 2), 0, math.sin(PITCH / 2), 0),
      joint_pos=JOINT_POSE,
    ),
    articulation=PIKACHU_V033_ARTICULATION,
    collisions=(
      CollisionCfg(
        geom_names_expr=(SUPPORT_GEOMS, "crawl_torso_collision"),
        contype=1,
        conaffinity=1,
        condim=3,
        priority=1,
        friction=(0.8,),
      ),
    ),
  )
