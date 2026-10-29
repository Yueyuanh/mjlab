"""Local V033 policy coordinates over V034's unchanged native articulation.

The hip roll/yaw bases differ by a rotation as well as a sign. Match their
angular Jacobians at the default pose; this is a local coordinate map, not an
exact kinematic equivalence between the two robots at arbitrary joint angles.
"""

from functools import lru_cache

import mujoco
import numpy as np
from scipy.optimize import least_squares

from mjlab.asset_zoo.robots.pikachu.pikachu_v033_constants import (
  get_pikachu_v033_robot_cfg,
)
from mjlab.asset_zoo.robots.pikachu.pikachu_v034_constants import (
  POLICY_ACTION_JOINT_NAMES,
  POLICY_OBSERVATION_JOINT_NAMES,
  get_biped_robot_cfg,
)
from mjlab.entity import Entity


@lru_cache(maxsize=1)
def coordinate_maps() -> dict[str, dict[str, float]]:
  """Return native-joint rows of the reference-to-native coordinate map."""
  models = []
  for factory in (get_pikachu_v033_robot_cfg, get_biped_robot_cfg):
    model = Entity(factory()).compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    models.append((model, data))
  (reference, ref_data), (native, native_data) = models
  rows = {}
  for i in range(reference.njnt):
    joint = reference.joint(i)
    if joint.type[0] != mujoco.mjtJoint.mjJNT_HINGE:
      continue
    j = native.joint(joint.name).id
    dot = float(ref_data.xaxis[i] @ native_data.xaxis[j])
    if abs(dot) < 0.7:
      raise ValueError(f"Unexpected axis mismatch for {joint.name}: {dot}")
    rows[joint.name] = {joint.name: 1.0 if dot > 0 else -1.0}
  for side in ("right", "left"):
    names = [f"{side}_hip_{kind}_joint" for kind in ("roll", "yaw")]
    ref_axes = np.stack([ref_data.xaxis[reference.joint(n).id] for n in names], -1)
    native_axes = np.stack([native_data.xaxis[native.joint(n).id] for n in names], -1)
    transform = np.linalg.lstsq(native_axes, ref_axes, rcond=None)[0]
    np.testing.assert_allclose(native_axes @ transform, ref_axes, atol=1e-6)
    for name, row in zip(names, transform, strict=True):
      rows[name] = dict(zip(names, map(float, row), strict=True))
  return rows


def coordinate_matrix(names: tuple[str, ...]) -> np.ndarray:
  rows = coordinate_maps()
  return np.array([[rows[name].get(other, 0.0) for other in names] for name in names])


def _closed_chain_jacobians(factory):
  model = Entity(factory()).compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
  result = {}
  for side in ("right", "left"):
    names = [f"{side}_{kind}_joint" for kind in ("knee_up", "knee", "knee_mid")]
    qids = [model.joint(name).qposadr[0] for name in names]

    def residual(values, qids=qids, side=side):
      data.qpos[qids[1:]] = values
      mujoco.mj_forward(model, data)
      return (
        data.site(f"{side}_knee_mid_closure").xpos - data.site(f"{side}_knee_mid").xpos
      )

    data.qpos[qids[0]] = 0
    fit = least_squares(residual, [0.0, 0.0], xtol=1e-14, ftol=1e-14, gtol=1e-14)
    zero = fit.x.copy()
    if np.linalg.norm(fit.fun) > 1e-8:
      raise ValueError(f"Cannot close the {side} reference linkage")
    residual(zero)
    position, rotation = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, position, rotation, model.site(f"{side}_foot").id)
    jacobian = np.concatenate((position[[0, 2]], rotation[[1]]))
    epsilon = 1e-5
    data.qpos[qids[0]] = epsilon
    fit = least_squares(residual, zero, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    slopes = (fit.x - zero) / epsilon
    columns = []
    for kind in ("hip_pitch", "knee_up", "ankle"):
      column = jacobian[:, model.joint(f"{side}_{kind}_joint").dofadr[0]].copy()
      if kind == "knee_up":
        for passive, slope in zip(names[1:], slopes, strict=True):
          column += jacobian[:, model.joint(passive).dofadr[0]] * slope
      columns.append(column)
    result[side] = (np.stack(columns, -1), zero, slopes)
    data.qpos[qids[0]] = 0
    residual(zero)
  return result


@lru_cache(maxsize=1)
def foot_coordinate_matrices():
  """Match local foot x/z/pitch motion, including both passive knee joints.

  Observation offsets represent the reference linkage's closed rest state.
  Neither robot's XML nor its native reset pose is modified.
  """
  actions, states = POLICY_ACTION_JOINT_NAMES, POLICY_OBSERVATION_JOINT_NAMES
  reference = _closed_chain_jacobians(get_pikachu_v033_robot_cfg)
  native = _closed_chain_jacobians(get_biped_robot_cfg)
  action_map = coordinate_matrix(actions)
  for side in ("right", "left"):
    ids = [
      actions.index(f"{side}_{kind}_joint")
      for kind in ("hip_pitch", "knee_up", "ankle")
    ]
    action_map[np.ix_(ids, ids)] = np.linalg.solve(native[side][0], reference[side][0])
  inverse = np.linalg.inv(action_map)
  state_map = np.zeros((len(states), len(states)))
  state_bias = np.zeros(len(states))
  driven_ids = [states.index(name) for name in actions]
  state_map[np.ix_(driven_ids, driven_ids)] = inverse
  for side in ("right", "left"):
    knee = states.index(f"{side}_knee_up_joint")
    for i, kind in enumerate(("knee", "knee_mid")):
      name = f"{side}_{kind}_joint"
      index = states.index(name)
      sign = coordinate_maps()[name][name]
      state_map[index] = reference[side][2][i] * state_map[knee]
      state_map[index, index] += sign
      state_map[index, knee] -= sign * native[side][2][i]
      state_bias[index] = reference[side][1][i] - sign * native[side][1][i]
  return action_map, state_map, state_bias
