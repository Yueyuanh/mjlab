"""Local V033 crawl coordinates over V034's native crawl articulation."""

from functools import lru_cache

import mujoco
import numpy as np
from scipy.optimize import least_squares

from mjlab.asset_zoo.robots.pikachu.pikachu_v034_constants import (
  POLICY_ACTION_JOINT_NAMES,
  POLICY_OBSERVATION_JOINT_NAMES,
)
from mjlab.entity import Entity
from mjlab.tasks.velocity.config.pikachu_v033_crawl.robot import (
  robot_cfg as v033_crawl_robot_cfg,
)
from mjlab.tasks.velocity.config.pikachu_v034.robot import get_crawl_robot_cfg

_LEG_KINDS = ("hip_pitch", "hip_roll", "hip_yaw", "knee_up", "ankle")
_ARM_KINDS = ("arm_pitch", "arm_roll", "arm_yaw", "elbow")
_PASSIVE_KINDS = ("knee", "knee_mid")
_ANGULAR_LENGTH_SCALE = 0.10


def _model_and_data(factory):
  model = Entity(factory()).compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
  mujoco.mj_forward(model, data)
  return model, data


def _passive_slopes(model, data, side: str) -> np.ndarray:
  """Differentiate the closed-chain passive knee coordinates at crawl pose."""
  names = [f"{side}_{kind}_joint" for kind in ("knee_up", *_PASSIVE_KINDS)]
  qids = [model.joint(name).qposadr[0] for name in names]
  baseline = data.qpos.copy()
  epsilon = 1e-5

  def residual(values):
    data.qpos[:] = baseline
    data.qpos[qids[0]] += epsilon
    data.qpos[qids[1:]] = values
    mujoco.mj_forward(model, data)
    return data.site(f"{side}_knee_mid_closure").xpos - data.site(
      f"{side}_knee_mid"
    ).xpos

  fit = least_squares(
    residual,
    baseline[qids[1:]],
    xtol=1e-14,
    ftol=1e-14,
    gtol=1e-14,
  )
  if np.linalg.norm(fit.fun) > 1e-8:
    raise ValueError(f"Cannot differentiate the {side} crawl linkage")
  data.qpos[:] = baseline
  mujoco.mj_forward(model, data)
  return (fit.x - baseline[qids[1:]]) / epsilon


def _foot_jacobian(model, data, side: str) -> tuple[np.ndarray, np.ndarray]:
  """Return base-frame foot twist Jacobian including passive knee motion."""
  position = np.zeros((3, model.nv))
  rotation = np.zeros((3, model.nv))
  mujoco.mj_jacSite(
    model, data, position, rotation, model.site(f"{side}_foot").id
  )
  base_rotation = data.xmat[model.body("base_link").id].reshape(3, 3).T
  slopes = _passive_slopes(model, data, side)

  columns = []
  for kind in _LEG_KINDS:
    dof = model.joint(f"{side}_{kind}_joint").dofadr[0]
    linear = base_rotation @ position[:, dof]
    angular = base_rotation @ rotation[:, dof]
    column = np.r_[linear, _ANGULAR_LENGTH_SCALE * angular]
    if kind == "knee_up":
      for passive_kind, slope in zip(_PASSIVE_KINDS, slopes, strict=True):
        passive_dof = model.joint(f"{side}_{passive_kind}_joint").dofadr[0]
        passive = np.r_[
          base_rotation @ position[:, passive_dof],
          _ANGULAR_LENGTH_SCALE * (base_rotation @ rotation[:, passive_dof]),
        ]
        column += slope * passive
    columns.append(column)
  return np.stack(columns, axis=-1), slopes


def _axis_sign(reference, native, name: str) -> float:
  reference_axis = reference.jnt_axis[reference.joint(name).id]
  native_axis = native.jnt_axis[native.joint(name).id]
  dot = float(reference_axis @ native_axis)
  if abs(dot) < 0.99:
    raise ValueError(f"Unexpected local-axis mismatch for {name}: {dot}")
  return 1.0 if dot > 0 else -1.0


@lru_cache(maxsize=1)
def crawl_coordinate_matrices() -> tuple[np.ndarray, np.ndarray]:
  """Return reference-to-native actions and native-to-reference observations.

  Legs are mapped by their local foot twist at the calibrated crawl poses.  The
  arm endpoint Jacobian is rank deficient near the extended support pose, so arm
  coordinates use the exact local joint-axis signs instead of an unstable inverse.
  """
  reference, ref_data = _model_and_data(v033_crawl_robot_cfg)
  native, native_data = _model_and_data(get_crawl_robot_cfg)
  actions = POLICY_ACTION_JOINT_NAMES
  states = POLICY_OBSERVATION_JOINT_NAMES
  action_map = np.zeros((len(actions), len(actions)))
  passive_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

  for side in ("right", "left"):
    reference_jacobian, reference_slopes = _foot_jacobian(
      reference, ref_data, side
    )
    native_jacobian, native_slopes = _foot_jacobian(native, native_data, side)
    leg_map = np.linalg.lstsq(native_jacobian, reference_jacobian, rcond=None)[0]
    if np.linalg.cond(leg_map) > 10:
      raise ValueError(f"Ill-conditioned {side} crawl leg mapping")
    relative_error = np.linalg.norm(native_jacobian @ leg_map - reference_jacobian)
    relative_error /= np.linalg.norm(reference_jacobian)
    if relative_error > 0.08:
      raise ValueError(f"Poor {side} crawl foot mapping: {relative_error}")
    ids = [actions.index(f"{side}_{kind}_joint") for kind in _LEG_KINDS]
    action_map[np.ix_(ids, ids)] = leg_map
    passive_data[side] = reference_slopes, native_slopes

    for kind in _ARM_KINDS:
      name = f"{side}_{kind}_joint"
      index = actions.index(name)
      action_map[index, index] = _axis_sign(reference, native, name)

  inverse = np.linalg.inv(action_map)
  observation_map = np.zeros((len(states), len(states)))
  driven_ids = [states.index(name) for name in actions]
  observation_map[np.ix_(driven_ids, driven_ids)] = inverse

  # Preserve any measured constraint error while expressing passive knees in
  # the reference linkage coordinates around each calibrated crawl pose.
  for side in ("right", "left"):
    knee = states.index(f"{side}_knee_up_joint")
    reference_slopes, native_slopes = passive_data[side]
    for i, kind in enumerate(_PASSIVE_KINDS):
      name = f"{side}_{kind}_joint"
      index = states.index(name)
      sign = _axis_sign(reference, native, name)
      observation_map[index] = reference_slopes[i] * observation_map[knee]
      observation_map[index, index] += sign
      observation_map[index, knee] -= sign * native_slopes[i]
  return action_map, observation_map
