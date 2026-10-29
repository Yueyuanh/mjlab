import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
folder = args.folder
args.output.parent.mkdir(parents=True, exist_ok=True)
meta = json.loads((folder / "rollouts.json").read_text())
a = np.load(folder / "rollouts.npz")
m = mujoco.MjModel.from_binary_path(str(folder / "compiled_model.mjb"))
d = mujoco.MjData(m)


def named(kind, suffix):
  n = {"body": m.nbody, "joint": m.njnt, "site": m.nsite, "geom": m.ngeom}[kind]
  return next(i for i in range(n) if getattr(m, kind)(i).name.split("/")[-1] == suffix)


sides = ["right", "left"]
body = [named("body", f"{s}_ankle_link") for s in sides]
joints = [named("joint", f"{s}_ankle_joint") for s in sides]
sites = [named("site", f"{s}_foot") for s in sides]
contactids = [
  next(i for i, n in enumerate(meta["contact_names"]) if n.endswith(f"{s}_ankle_link"))
  for s in sides
]
alljointids = [meta["all_joint_names"].index(f"{s}_ankle_joint") for s in sides]
actids = [next(i for i in range(m.nu) if m.actuator_trnid[i, 0] == j) for j in joints]
compiled = {}
for side, b, j in zip(sides, body, joints, strict=False):
  compiled[side] = {
    "ankle_limit_deg": np.rad2deg(m.jnt_range[j]).tolist(),
    "axis": m.jnt_axis[j].tolist(),
    "actuator_force_limit": m.actuator_forcerange[actids[sides.index(side)]].tolist(),
    "geoms": [],
  }
  for g in range(m.ngeom):
    if m.geom_bodyid[g] == b:
      compiled[side]["geoms"].append(
        {
          "name": m.geom(g).name,
          "type": int(m.geom_type[g]),
          "contype": int(m.geom_contype[g]),
          "conaffinity": int(m.geom_conaffinity[g]),
          "size": m.geom_size[g].tolist(),
          "friction": m.geom_friction[g].tolist(),
          "condim": int(m.geom_condim[g]),
        }
      )
results = []
for case in meta["cases"]:
  key = case["key"]
  qpos = a[f"{key}/qpos"]
  start = round(meta["warmup_s"] / meta["dt"])
  steps, n = qpos.shape[:2]
  tilt = np.zeros((steps, n, 2))
  pitch = tilt.copy()
  height = tilt.copy()
  gap = tilt.copy()
  flatq = tilt.copy()
  ankle = tilt.copy()
  flatresid = tilt.copy()
  for t in range(steps):
    for env in range(n):
      d.qpos[:] = qpos[t, env]
      mujoco.mj_kinematics(m, d)
      for k, (b, j, site) in enumerate(zip(body, joints, sites, strict=False)):
        R = d.xmat[b].reshape(3, 3)
        tilt[t, env, k] = np.rad2deg(np.arccos(np.clip(R[2, 2], -1, 1)))
        pitch[t, env, k] = np.rad2deg(np.arctan2(-R[2, 0], R[2, 2]))
        height[t, env, k] = d.site_xpos[site, 2]
        gap[t, env, k] = -0.08 * R[2, 0]
        q = d.qpos[m.jnt_qposadr[j]]
        ankle[t, env, k] = q
        flatq[t, env, k] = q + np.arctan2(R[2, 0], R[2, 2]) / m.jnt_axis[j, 1]
        flatresid[t, env, k] = np.rad2deg(
          np.arccos(np.clip(np.hypot(R[2, 0], R[2, 2]), 0, 1))
        )
  contact = a[f"{key}/contact"][..., contactids] > 0
  forces = a[f"{key}/force"][..., 2]
  result = {
    **case,
    "velocity_mean": a[f"{key}/velocity"][start:].mean((0, 1)).tolist(),
    "falls": int(a[f"{key}/terminated"].sum()),
    "rear_force_share": float(
      np.abs(forces[start:, ..., contactids]).sum() / np.abs(forces[start:]).sum()
    ),
    "feet": {},
    "reward_mean": dict(
      zip(
        meta["reward_names"],
        a[f"{key}/rewards"][start:].mean((0, 1)).tolist(),
        strict=False,
      )
    ),
  }
  for k, (s, j) in enumerate(zip(sides, joints, strict=False)):
    mask = contact[start:, :, k]
    lo, hi = m.jnt_range[j]
    q = ankle[start:, :, k]
    flat = flatq[start:, :, k]
    trg = a[f"{key}/target"][start:, :, alljointids[k]]

    def avg(v, mask=mask):
      return float(np.mean(v[mask]))

    result["feet"][s] = {
      "contact_fraction": float(mask.mean()),
      "stance_tilt_mean_deg": avg(tilt[start:, :, k]),
      "stance_tilt_p95_deg": float(np.percentile(tilt[start:, :, k][mask], 95)),
      "stance_pitch_mean_deg": avg(pitch[start:, :, k]),
      "stance_over10deg_fraction": avg(tilt[start:, :, k] > 10),
      "stance_toe_lower_10mm_fraction": avg(gap[start:, :, k] > 0.01),
      "stance_heel_minus_toe_mean_mm": avg(gap[start:, :, k]) * 1000,
      "stance_center_height_mean_mm": avg(height[start:, :, k]) * 1000,
      "ankle_mean_deg": avg(np.rad2deg(q)),
      "ankle_range_deg": np.rad2deg([q.min(), q.max()]).tolist(),
      "ankle_near_limit_3deg_fraction": avg(np.minimum(q - lo, hi - q) < np.deg2rad(3)),
      "flat_ankle_mean_deg": avg(np.rad2deg(flat)),
      "flat_ankle_outside_limits_fraction": avg((flat < lo) | (flat > hi)),
      "flat_remaining_roll_mean_deg": avg(flatresid[start:, :, k]),
      "ankle_target_error_abs_mean_deg": avg(np.rad2deg(np.abs(trg - q))),
      "ankle_torque_abs_mean_nm": avg(
        np.abs(a[f"{key}/actuator_force"][start:, :, actids[k]])
      ),
      "ankle_torque_over12nm_fraction": avg(
        np.abs(a[f"{key}/actuator_force"][start:, :, actids[k]]) > 12
      ),
    }
  # Distinguish edge contact at transitions from loaded/middle stance.
  for k, s in enumerate(sides):
    c = contact[..., k]
    middle = np.zeros_like(c)
    for e in range(n):
      edges = np.diff(np.r_[False, c[:, e], False].astype(int))
      starts = np.where(edges == 1)[0]
      ends = np.where(edges == -1)[0]
      for st, en in zip(starts, ends, strict=False):
        cut = max(1, int((en - st) * 0.2))
        middle[st + cut : en - cut, e] = True
    load = (c & (np.abs(forces[..., contactids[k]]) > 5))[start:]
    central = middle[start:]
    result["feet"][s]["loaded_over5N_tilt_mean_deg"] = float(
      tilt[start:, :, k][load].mean()
    )
    result["feet"][s]["middle_stance_tilt_mean_deg"] = float(
      tilt[start:, :, k][central].mean()
    )
    result["feet"][s]["middle_stance_over10deg_fraction"] = float(
      (tilt[start:, :, k][central] > 10).mean()
    )
  arrays = {
    f"{key}/tilt": tilt,
    f"{key}/pitch": pitch,
    f"{key}/height": height,
    f"{key}/flatq": flatq,
    f"{key}/gap": gap,
  }
  # Keep derived samples for plots and follow-up diagnostics.
  if "derived" not in globals():
    derived = {}
  derived.update(arrays)
  results.append(result)
report = {
  "protocol": meta,
  "compiled_model": compiled,
  "cases": results,
  "geometry_note": "Sole normal is ankle local +Z. Toe label uses local +X, heel -X; centerline capsule endpoints span 80 mm. Footprint and normal statistics use actual world orientation. Contact is simulation body sensor, not inferred from height. Excludes first 2 seconds. Flat-ankle estimate holds all other joints fixed; it is a kinematic feasibility probe, not a rollout.",
}
args.output.write_text(json.dumps(report, indent=2) + "\n")
np.savez_compressed(folder / "derived.npz", **derived)
for r in results:
  print(r["key"], "vx", round(r["velocity_mean"][0], 3), "falls", r["falls"])
  for s, v in r["feet"].items():
    print(
      s,
      {
        k: round(v[k], 2)
        for k in [
          "stance_tilt_mean_deg",
          "stance_pitch_mean_deg",
          "stance_heel_minus_toe_mean_mm",
          "ankle_mean_deg",
          "flat_ankle_mean_deg",
          "flat_ankle_outside_limits_fraction",
          "ankle_near_limit_3deg_fraction",
          "ankle_target_error_abs_mean_deg",
          "ankle_torque_abs_mean_nm",
        ]
      },
    )
