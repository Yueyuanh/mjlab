"""Solve a symmetric V035 four-support crawl pose (direct-drive knee).

V035 has no four-bar closure, so the pose is solved only from the four
support targets (both soles flat + both hands on the ground).  Joint angles
are mirrored by ``right = -left`` (the native axes of the two sides are
opposite or equal in a way that always requires this sign for a symmetric
pose).  Bounds use the intersection of the native left/right ranges.

Run from the mjlab repository root::

    uv run python scripts/calibrate_v035_crawl.py
"""

import json

import mujoco
import numpy as np
from scipy.optimize import least_squares

from mjlab.asset_zoo.robots.pikachu.pikachu_v035_constants import ASSET_DIR, get_spec

PITCH = 1.4  # 躯干前倾 ~80 deg
HAND_RADIUS = 0.016
KINDS = (
  "hip_pitch",
  "hip_roll",
  "hip_yaw",
  "knee",
  "ankle",
  "arm_pitch",
  "arm_roll",
  "arm_yaw",
  "elbow",
)
# 与 v034 相同的支撑目标（m，世界系相对 base）：脚在身后、手在身前。
FOOT_TARGET = np.array([-0.09, -0.10, 0.0])
HAND_TARGET = np.array([0.24, -0.13, HAND_RADIUS])


def main():
  spec = get_spec()
  for side, sign in (("left", 1), ("right", -1)):
    spec.body(f"{side}_elbow_link").add_site(
      name=f"{side}_hand", pos=[0, sign * 0.060, 0], size=[0.003]
    )
  model = spec.compile()
  data = mujoco.MjData(model)
  low, high = [], []
  for kind in KINDS:
    right, left = model.joint(f"right_{kind}_joint"), model.joint(f"left_{kind}_joint")
    if right.limited:
      low.append(max(right.range[0], -left.range[1]) + 0.03)
      high.append(min(right.range[1], -left.range[0]) - 0.03)
    else:
      low.append(-2.5)
      high.append(2.5)
  low, high = np.array(low), np.array(high)
  low = np.append(low, 0.13)
  high = np.append(high, 0.30)

  def pose(x):
    data.qpos[:] = model.qpos0
    data.qpos[:7] = [0, 0, x[-1], np.cos(PITCH / 2), 0, np.sin(PITCH / 2), 0]
    for side, sign in (("right", 1), ("left", -1)):
      for kind, value in zip(KINDS, x[:-1], strict=True):
        data.qpos[model.joint(f"{side}_{kind}_joint").qposadr[0]] = sign * value
    mujoco.mj_forward(model, data)

  def residual(x):
    pose(x)
    foot, hand = data.site("right_foot").xpos, data.site("right_hand").xpos
    normal = data.xmat[model.body("right_ankle_link").id].reshape(3, 3)[:, 2]
    return np.concatenate(
      (
        (foot - FOOT_TARGET) * [5, 10, 200],
        (hand - HAND_TARGET) * [5, 10, 200],
        (normal - [0, 0, 1]) * 4,
        x[:-1] * 0.002,
      )
    )

  best = None
  for seed in range(12):
    x = np.random.default_rng(seed).uniform(low, high)
    fit = least_squares(residual, x, bounds=(low, high), max_nfev=2000)
    if best is None or np.linalg.norm(fit.fun) < np.linalg.norm(best.fun):
      best = fit
  assert best is not None
  pose(best.x)
  result = {
    "pitch": PITCH,
    "height": float(best.x[-1]),
    "hand_radius": HAND_RADIUS,
    "joints": {
      model.joint(i).name: float(data.qpos[model.jnt_qposadr[i]])
      for i in range(1, model.njnt)
    },
    "supports": {
      f"{side}_{kind}": data.site(f"{side}_{kind}").xpos.tolist()
      for side in ("right", "left")
      for kind in ("foot", "hand")
    },
    "com": data.subtree_com[model.body("base_link").id].tolist(),
    "residual_norm": float(np.linalg.norm(best.fun)),
  }
  feet_z = [abs(result["supports"][f"{s}_foot"][2]) for s in ("left", "right")]
  hands_z = [abs(result["supports"][f"{s}_hand"][2] - HAND_RADIUS) for s in ("left", "right")]
  print(json.dumps(result, indent=2))
  assert max(feet_z) < 0.002, feet_z
  assert max(hands_z) < 0.003, hands_z
  (ASSET_DIR / "crawl_pose.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
  main()
