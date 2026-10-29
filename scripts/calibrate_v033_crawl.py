"""Solve the V033 crawl reset with symmetric IK and four-bar closure."""

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from mjlab.asset_zoo.robots.pikachu.pikachu_v033_constants import get_spec

OUT = Path("doc/pikachu_v033_experiments_20261019/config_audit")


def main():
  spec = get_spec()
  for side, sign in (("right", -1), ("left", 1)):
    spec.body(f"{side}_elbow_link").add_site(
      name=f"{side}_hand", pos=[0, sign * 0.09, 0], size=[0.01]
    )
  model = spec.compile()
  data = mujoco.MjData(model)
  kinds = (
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
  signs = (-1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1)

  def set_pose(x):
    data.qpos[:3] = [0, 0, x[-1]]
    data.qpos[3:7] = [np.cos(1.3 / 2), 0, np.sin(1.3 / 2), 0]
    for side, factors in (("right", np.ones(11)), ("left", signs)):
      for kind, value, factor in zip(kinds, x[:11], factors, strict=True):
        data.qpos[model.joint(f"{side}_{kind}_joint").qposadr[0]] = value * factor
    mujoco.mj_forward(model, data)

  def residual(x):
    set_pose(x)
    foot = data.site("right_foot").xpos
    hand = data.site("right_hand").xpos
    closure = (
      data.site("right_knee_mid_closure").xpos - data.site("right_knee_mid").xpos
    )
    normal = data.xmat[model.body("right_ankle_link").id].reshape(3, 3)[:, 2]
    return np.concatenate(
      (
        (foot - [-0.10, -0.10, 0]) * 20,
        (hand - [0.24, -0.12, 0.015]) * 20,
        closure * 200,
        normal[:2] * 2,
        x[:11] * 0.005,
      )
    )

  lo = [-2.4, -0.24, -0.24, -0.60, -2, -0.60, -0.75, -1.55, -1.55, -1.55, -1.55, 0.12]
  hi = [2.4, 0.24, 0.24, 1.0, 2, 1.0, 1.0, 1.55, 1.55, 1.55, 1.55, 0.25]
  best = None
  for seed in range(12):
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi)
    fit = least_squares(residual, x, bounds=(lo, hi), max_nfev=1500)
    if best is None or np.linalg.norm(fit.fun) < np.linalg.norm(best.fun):
      best = fit
  assert best is not None
  set_pose(best.x)
  joints = {
    model.joint(i).name: float(data.qpos[model.jnt_qposadr[i]])
    for i in range(1, model.njnt)
  }
  result = {
    "pitch": 1.3,
    "height": float(best.x[-1]),
    "joints": joints,
    "residual_norm": float(np.linalg.norm(best.fun)),
    "supports": {
      s: data.site(s).xpos.tolist()
      for s in ("right_foot", "left_foot", "right_hand", "left_hand")
    },
    "closure_m": float(
      np.linalg.norm(
        data.site("right_knee_mid_closure").xpos - data.site("right_knee_mid").xpos
      )
    ),
    "com": data.subtree_com[model.body("base_link").id].tolist(),
  }
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "crawl_calibration.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
