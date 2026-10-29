"""Solve a symmetric V034 four-support pose with native axes and joint limits."""

import json

import mujoco
import numpy as np
from scipy.optimize import least_squares

from mjlab.asset_zoo.robots.pikachu.pikachu_v034_constants import ASSET_DIR, get_spec

PITCH = 1.4
HAND_RADIUS = 0.016
KINDS = (
  "hip_pitch",
  "hip_roll",
  "hip_yaw",
  "knee_up",
  "knee_mid",
  "knee",
  "ankle",
  "arm_pitch",
  "arm_roll",
  "arm_yaw",
  "elbow",
)


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
  low.append(0.13)
  high.append(0.30)

  def pose(x):
    data.qpos[:] = model.qpos0
    data.qpos[:7] = [0, 0, x[-1], np.cos(PITCH / 2), 0, np.sin(PITCH / 2), 0]
    for side, sign in (("right", 1), ("left", -1)):
      for kind, value in zip(KINDS, x[:11], strict=True):
        data.qpos[model.joint(f"{side}_{kind}_joint").qposadr[0]] = sign * value
    mujoco.mj_forward(model, data)

  def residual(x):
    pose(x)
    foot, hand = data.site("right_foot").xpos, data.site("right_hand").xpos
    closure = (
      data.site("right_knee_mid_closure").xpos - data.site("right_knee_mid").xpos
    )
    normal = data.xmat[model.body("right_ankle_link").id].reshape(3, 3)[:, 2]
    return np.concatenate(
      (
        (foot - [-0.09, -0.10, 0]) * [5, 10, 200],
        (hand - [0.24, -0.13, HAND_RADIUS]) * [5, 10, 200],
        closure * 1000,
        (normal - [0, 0, 1]) * 4,
        x[:11] * 0.002,
      )
    )

  best = None
  for seed in range(8):
    x = np.random.default_rng(seed).uniform(low, high)
    fit = least_squares(residual, x, bounds=(low, high), max_nfev=1000)
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
    "closure_m": {
      side: float(
        np.linalg.norm(
          data.site(f"{side}_knee_mid_closure").xpos
          - data.site(f"{side}_knee_mid").xpos
        )
      )
      for side in ("left", "right")
    },
    "com": data.subtree_com[model.body("base_link").id].tolist(),
    "residual_norm": float(np.linalg.norm(best.fun)),
  }
  assert max(result["closure_m"].values()) < 1e-5, result
  assert (
    max(abs(result["supports"][f"{s}_foot"][2]) for s in ("left", "right")) < 0.001
  ), result
  (ASSET_DIR / "crawl_pose.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
