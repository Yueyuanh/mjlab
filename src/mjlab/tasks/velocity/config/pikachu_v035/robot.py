"""V035 四足（crawl）姿态与支撑碰撞体。"""

import json
import math
from functools import partial

import mujoco

from mjlab.asset_zoo.robots.pikachu.pikachu_v035_constants import (
  ARTICULATION,
  ASSET_DIR,
  get_spec,
)
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

CALIBRATION = json.loads((ASSET_DIR / "crawl_pose.json").read_text())
HEIGHT, PITCH = CALIBRATION["height"], CALIBRATION["pitch"]
SUPPORT_SITES = ("right_foot", "left_foot", "right_hand", "left_hand")
SUPPORT_GEOMS = r"(left|right)_(foot[1-5]|hand)_collision"


def crawl_spec() -> mujoco.MjSpec:
  spec = get_spec()
  for side, sign in (("left", 1), ("right", -1)):
    body = spec.body(f"{side}_elbow_link")
    # 圆球近似前臂远端支撑点，位于 elbow STL 的 y 向范围内。
    position = [0, sign * 0.060, 0]
    body.add_site(name=f"{side}_hand", pos=position, size=[0.003])
    body.add_geom(
      name=f"{side}_hand_collision",
      pos=position,
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      size=[CALIBRATION["hand_radius"], 0, 0],
      mass=0,
      group=3,
    )
  # 躯干外包盒：与 base_link STL 的局部 AABB 一致（v034/v035 同一机壳）。
  spec.body("base_link").add_geom(
    name="torso_collision",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=[0.0414, 0.0015, 0.083],
    size=[0.068, 0.0495, 0.112],
    mass=0,
    group=3,
  )
  return spec


def get_crawl_robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=partial(crawl_spec),
    init_state=EntityCfg.InitialStateCfg(
      pos=(0, 0, HEIGHT + 0.003),
      rot=(math.cos(PITCH / 2), 0, math.sin(PITCH / 2), 0),
      joint_pos=CALIBRATION["joints"],
      joint_vel={".*": 0.0},
    ),
    articulation=ARTICULATION,
    collisions=(
      CollisionCfg(
        geom_names_expr=(SUPPORT_GEOMS, "torso_collision"),
        contype=1,
        conaffinity=1,
        condim=3,
        priority=1,
        friction=(0.8,),
      ),
    ),
  )
