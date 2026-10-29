"""Physical coordinate mapping and isolation of the V034 walking transfer."""

import mujoco
import numpy as np
from scipy.optimize import least_squares

from mjlab.asset_zoo.robots.pikachu.pikachu_v034_constants import (
  POLICY_ACTION_JOINT_NAMES,
  POLICY_OBSERVATION_JOINT_NAMES,
)
from mjlab.entity import Entity
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.pikachu_v034_biped_transfer.mapping import (
  coordinate_maps,
  coordinate_matrix,
  foot_coordinate_matrices,
)
from mjlab.tasks.velocity.config.pikachu_v034_biped_transfer.mdp import (
  TransferJointPositionActionCfg,
)

TASK = "Mjlab-Velocity-Flat-Pikachu-V034-Biped-Transfer"


def test_transfer_maps_hip_angular_jacobians_including_axis_rotation():
  axes = []
  names = POLICY_OBSERVATION_JOINT_NAMES
  for task in ("Mjlab-Velocity-Flat-Pikachu-V033-E1", TASK):
    cfg = load_env_cfg(task)
    model = Entity(cfg.scene.entities["robot"]).compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    axes.append(np.stack([data.xaxis[model.joint(n).id] for n in names], -1))
  transform = coordinate_matrix(names)
  np.testing.assert_allclose(axes[1] @ transform, axes[0], atol=1e-6)
  # CAD axis/quaternion rounding leaves a few microradians of nonorthogonality.
  np.testing.assert_allclose(transform.T @ transform, np.eye(len(names)), atol=1e-5)
  for side in ("left", "right"):
    # A sign-only adapter misses this coupling of approximately 42 degrees.
    assert (
      abs(coordinate_maps()[f"{side}_hip_roll_joint"][f"{side}_hip_yaw_joint"]) > 0.6
    )
  assert coordinate_maps()["right_knee_up_joint"] == {"right_knee_up_joint": -1.0}
  assert coordinate_maps()["left_knee_up_joint"] == {"left_knee_up_joint": 1.0}


def test_transfer_preserves_native_robot_and_has_distinct_policy_interface():
  cfg = load_env_cfg(TASK)
  original = load_env_cfg("Mjlab-Velocity-Flat-Pikachu-V034-Biped-E1")
  actual = Entity(cfg.scene.entities["robot"]).compile()
  expected = Entity(original.scene.entities["robot"]).compile()
  for name in (
    "body_mass",
    "body_inertia",
    "body_pos",
    "body_quat",
    "jnt_axis",
    "jnt_range",
    "dof_damping",
    "dof_armature",
    "actuator_gainprm",
    "actuator_forcerange",
    "geom_friction",
    "eq_data",
  ):
    np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
  assert isinstance(cfg.actions["joint_pos"], TransferJointPositionActionCfg)
  for group in cfg.observations.values():
    assert group.terms["joint_pos"].func.__name__ == "joint_pos_reference"
    assert group.terms["joint_vel"].func.__name__ == "joint_vel_reference"
  agent = load_rl_cfg(TASK)
  assert agent.algorithm.symmetry_cfg is None
  assert agent.wandb_project == "pikachu-v034-velocity-biped-transfer"


def test_foot_retarget_matches_closed_chain_foot_motion():
  action_map, observation_map, _ = foot_coordinate_matrices()
  names = POLICY_ACTION_JOINT_NAMES
  models = [
    Entity(load_env_cfg(task).scene.entities["robot"]).compile()
    for task in (
      "Mjlab-Velocity-Flat-Pikachu-V033-E1",
      "Mjlab-Velocity-Flat-Pikachu-V034-Biped-Retarget",
    )
  ]

  def foot(model, side, drives):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    for name, value in zip(names, drives, strict=True):
      data.qpos[model.joint(name).qposadr[0]] += value
    passive = [
      model.joint(f"{side}_{kind}_joint").qposadr[0] for kind in ("knee", "knee_mid")
    ]

    def closure(values):
      data.qpos[passive] = values
      mujoco.mj_forward(model, data)
      return (
        data.site(f"{side}_knee_mid_closure").xpos - data.site(f"{side}_knee_mid").xpos
      )

    fit = least_squares(closure, [0.0, 0.0], xtol=1e-14, ftol=1e-14, gtol=1e-14)
    assert np.linalg.norm(closure(fit.x)) < 1e-8
    position = data.site(f"{side}_foot").xpos[[0, 2]]
    rotation = data.site(f"{side}_foot").xmat.reshape(3, 3)
    return np.r_[position, np.arctan2(rotation[0, 2], rotation[2, 2])]

  for side in ("right", "left"):
    baselines = [foot(model, side, np.zeros(len(names))) for model in models]
    for kind in ("hip_pitch", "knee_up", "ankle"):
      reference = np.zeros(len(names))
      reference[names.index(f"{side}_{kind}_joint")] = 1e-3
      native = action_map @ reference
      actual_delta = foot(models[1], side, native) - baselines[1]
      expected_delta = foot(models[0], side, reference) - baselines[0]
      np.testing.assert_allclose(actual_delta, expected_delta, atol=2e-6, rtol=0.01)
  # State and action transforms share the same driven-coordinate interface.
  ids = [POLICY_OBSERVATION_JOINT_NAMES.index(name) for name in names]
  np.testing.assert_allclose(
    observation_map[np.ix_(ids, ids)] @ action_map, np.eye(len(names)), atol=1e-10
  )


def test_load_adapted_gains_keep_native_inertia_and_motor_limits():
  native_cfg = load_env_cfg(TASK)
  load_cfg = load_env_cfg(TASK + "-Load")
  native = Entity(native_cfg.scene.entities["robot"]).compile()
  adapted = Entity(load_cfg.scene.entities["robot"]).compile()
  for name in (
    "body_mass",
    "body_inertia",
    "dof_armature",
    "jnt_axis",
    "jnt_range",
    "actuator_forcerange",
    "eq_data",
  ):
    np.testing.assert_array_equal(getattr(native, name), getattr(adapted, name))
  np.testing.assert_allclose(
    adapted.actuator_gainprm[:, 0], 1.9 * native.actuator_gainprm[:, 0]
  )
  # Loading another task must not mutate the shared native robot constants.
  rebuilt = Entity(load_env_cfg(TASK).scene.entities["robot"]).compile()
  np.testing.assert_array_equal(native.actuator_gainprm, rebuilt.actuator_gainprm)
