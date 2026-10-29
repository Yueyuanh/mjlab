"""Small CPU feasibility study: free-floating PD transitions, never root forcing.

Compare direct pose interpolation against closed-chain/support-aware IK. This
is a diagnostic, not a trained transition policy or a reference certification.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import least_squares

from mjlab.entity import Entity
from mjlab.tasks.velocity.config.pikachu_v033_crawl.robot import (
  PITCH,
  robot_cfg,
)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  if (args.out / "results.json").exists():
    raise FileExistsError(args.out / "results.json")
  entity = Entity(robot_cfg())
  entity.spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[10, 10, 0.1],
    contype=1,
    conaffinity=1,
    friction=[0.8, 0.005, 0.0001],
  )
  model = entity.compile()
  model.opt.timestep = 0.002
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
  quad = data.qpos.copy()
  biped = quad.copy()
  biped[:7] = [0, 0, 0.259412, 1, 0, 0, 0]
  biped[7:] = 0
  for side, sign in (("right", 1), ("left", -1)):
    biped[model.joint(f"{side}_arm_roll_joint").qposadr[0]] = sign * math.radians(53)
  sites = [
    model.site(n).id for n in ("right_foot", "left_foot", "right_hand", "left_hand")
  ]
  closure_sites = [
    (model.site(f"{side}_knee_mid_closure").id, model.site(f"{side}_knee_mid").id)
    for side in ("right", "left")
  ]
  root = model.body("base_link").id
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
  right = np.array([model.joint(f"right_{k}_joint").qposadr[0] for k in kinds])
  left = np.array([model.joint(f"left_{k}_joint").qposadr[0] for k in kinds])
  signs = np.array([-1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1])
  low, high = [], []
  for kind in kinds:
    joint = model.joint(f"right_{kind}_joint")
    low.append(joint.range[0] if joint.limited[0] else -math.pi)
    high.append(joint.range[1] if joint.limited[0] else math.pi)
  low += [-0.3, 0.13]
  high += [0.3, 0.32]

  def fk(qpos):
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)

  def unpack(x, phase):
    q = biped.copy()
    q[0], q[2] = x[-2:]
    q[3:7] = [math.cos(PITCH * phase / 2), 0, math.sin(PITCH * phase / 2), 0]
    q[right], q[left] = x[:11], x[:11] * signs
    return q

  def naive(phase):
    q = biped * (1 - phase) + quad * phase
    q[3:7] = [math.cos(PITCH * phase / 2), 0, math.sin(PITCH * phase / 2), 0]
    return q

  fk(biped)
  initial_feet = data.site_xpos[sites[:2]].copy()
  phases = np.linspace(0, 1, 61)
  direct = np.stack([naive(p) for p in phases])
  corrected = []
  previous = np.r_[biped[right], biped[[0, 2]]]
  for phase in phases:
    reference = naive(phase)
    # Move the support polygon smoothly from biped to crawl, rather than
    # allowing foot lift caused by unconstrained joint interpolation.
    fk(quad)
    feet_target = initial_feet * (1 - phase) + data.site_xpos[sites[:2]] * phase
    hand_target = data.site_xpos[sites[2:]].copy()
    hand_target[:, 2] += 0.12 * (1 - phase)

    def residual(
      x,
      phase=phase,
      feet_target=feet_target,
      hand_target=hand_target,
      reference=reference,
      previous=previous,
    ):
      fk(unpack(x, phase))
      closure = np.concatenate(
        [data.site_xpos[a] - data.site_xpos[b] for a, b in closure_sites]
      )
      feet = data.site_xpos[sites[:2]]
      hands = data.site_xpos[sites[2:]]
      # A quasi-static COM condition is a useful screen; it is not sufficient
      # for dynamic balance or for a successful handoff to locomotion.
      front = np.max(feet[:, 0]) * (1 - phase) + np.max(hands[:, 0]) * phase
      rear = np.min(feet[:, 0]) - 0.04
      com_x = data.subtree_com[root, 0]
      com_excess = max(rear - com_x, 0) + max(com_x - front - 0.04, 0)
      return np.r_[
        closure * 500,
        (feet - feet_target).ravel() * 30,
        (hands - hand_target).ravel() * 5,
        com_excess * 20,
        (x[:11] - reference[right]) * 0.08,
        (x - previous) * 0.05,
      ]

    guess = np.clip(previous, np.array(low) + 1e-8, np.array(high) - 1e-8)
    fit = least_squares(residual, guess, bounds=(low, high), max_nfev=100)
    previous = fit.x.copy()
    corrected.append(unpack(fit.x, phase))
  corrected = np.asarray(corrected)

  def geometric_metrics(sequence):
    closures, foot_z, hand_z, minimum_distances = [], [], [], []
    for q in sequence:
      fk(q)
      closures.append(
        max(
          np.linalg.norm(data.site_xpos[a] - data.site_xpos[b])
          for a, b in closure_sites
        )
      )
      foot_z.append(data.site_xpos[sites[:2], 2].copy())
      hand_z.append(data.site_xpos[sites[2:], 2].copy())
      minimum_distances.append(
        min(
          np.linalg.norm(data.site_xpos[sites[0]] - data.site_xpos[sites[2]]),
          np.linalg.norm(data.site_xpos[sites[1]] - data.site_xpos[sites[3]]),
        )
      )
    return {
      "max_four_bar_error_m": float(np.max(closures)),
      "max_abs_foot_height_m": float(np.abs(foot_z).max()),
      "min_hand_site_height_m": float(np.min(hand_z)),
      "min_same_side_endpoint_distance_m": float(np.min(minimum_distances)),
    }

  actuated_qpos = model.jnt_qposadr[model.actuator_trnid[:, 0]]

  def rollout(sequence, duration, direction, seed):
    mujoco.mj_resetData(model, data)
    rng = np.random.default_rng(seed)
    data.qpos[:] = sequence[0]
    # Small root velocity perturbations; joints/root are never overwritten
    # after this initialization. Actuator forces retain physical limits.
    data.qvel[:2] = rng.uniform(-0.01, 0.01, 2)
    mujoco.mj_forward(model, data)
    low_z, torso_contact, peak_tracking, peak_closure = 10.0, 0, 0.0, 0.0
    for step in range(int((duration + 2) / model.opt.timestep)):
      phase = min(step * model.opt.timestep / duration, 1)
      f = phase * (len(sequence) - 1)
      i = min(int(f), len(sequence) - 2)
      target = sequence[i] * (1 - (f - i)) + sequence[i + 1] * (f - i)
      data.ctrl[:] = target[actuated_qpos]
      mujoco.mj_step(model, data)
      low_z = min(low_z, data.qpos[2])
      torso_contact += sum(
        model.geom_bodyid[c.geom1] == root or model.geom_bodyid[c.geom2] == root
        for c in data.contact
      )
      peak_tracking = max(
        peak_tracking, np.abs(data.qpos[actuated_qpos] - target[actuated_qpos]).max()
      )
      peak_closure = max(
        peak_closure,
        max(
          np.linalg.norm(data.site_xpos[a] - data.site_xpos[b])
          for a, b in closure_sites
        ),
      )
      if not np.isfinite(data.qpos).all():
        raise RuntimeError("Nonfinite transition state")
    pitch = math.atan2(
      -data.xmat[root].reshape(3, 3)[2, 0], data.xmat[root].reshape(3, 3)[2, 2]
    )
    expected = PITCH if direction == "down" else 0.0
    success = (
      abs(pitch - expected) < 0.20
      and low_z > 0.10
      and torso_contact == 0
      and np.linalg.norm(data.qvel[:3]) < 0.1
    )
    return {
      "direction": direction,
      "duration_s": duration,
      "seed": seed,
      "success": bool(success),
      "final_pitch_rad": pitch,
      "final_height_m": float(data.qpos[2]),
      "minimum_height_m": float(low_z),
      "torso_contact_samples": int(torso_contact),
      "peak_joint_target_error_rad": float(peak_tracking),
      "peak_four_bar_error_m": float(peak_closure),
      "final_root_speed_m_s": float(np.linalg.norm(data.qvel[:3])),
    }

  result: dict[str, Any] = {
    "scope": "CPU free-floating PD feasibility screen; no policy training, no root forcing",
    "joint_names": [model.joint(i).name for i in range(1, model.njnt)],
    "action_dimension": int(model.nu),
    "old_reference_joint_dimensions": {},
    "geometry": {},
    "rollouts": {},
  }
  for direction in ("Down", "Up"):
    path = Path(f"src/mjlab/mocap/npz/Pikachu-Transition-{direction}.npz")
    with np.load(path) as old:
      result["old_reference_joint_dimensions"][direction] = int(
        old["joint_pos"].shape[-1]
      )
  for name, sequence in (("direct", direct), ("support_ik", corrected)):
    result["geometry"][name] = geometric_metrics(sequence)
    result["rollouts"][name] = [
      rollout(
        sequence if direction == "down" else sequence[::-1], duration, direction, seed
      )
      for duration in (4.0, 8.0)
      for direction in ("down", "up")
      for seed in (0, 1)
    ]
    np.savez_compressed(
      args.out / f"{name}_candidate.npz",
      qpos=sequence,
      phase=phases,
      joint_names=np.array(result["joint_names"]),
    )
  (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
