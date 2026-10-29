"""Crawl-pose coordinate transfer from V033 to V034."""

import numpy as np

from mjlab.asset_zoo.robots.pikachu.pikachu_v034_constants import (
  POLICY_ACTION_JOINT_NAMES,
  POLICY_OBSERVATION_JOINT_NAMES,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.pikachu_v034_crawl_transfer.mapping import (
  _ARM_KINDS,
  _LEG_KINDS,
  crawl_coordinate_matrices,
)
from mjlab.tasks.velocity.config.pikachu_v034_crawl_transfer.mdp import (
  CrawlTransferJointPositionActionCfg,
)

TASK = "Mjlab-Velocity-Flat-Pikachu-V034-Crawl-Retarget"


def test_crawl_transfer_is_isolated_and_uses_reference_action_scale():
  cfg = load_env_cfg(TASK)
  action = cfg.actions["joint_pos"]
  assert isinstance(action, CrawlTransferJointPositionActionCfg)
  assert action.scale[r".*_knee_up_joint"] == 0.25
  assert action.scale[r".*_arm_yaw_joint"] == 0.20
  for group in cfg.observations.values():
    assert group.terms["joint_pos"].func.__name__ == "joint_pos_reference"
    assert group.terms["joint_vel"].func.__name__ == "joint_vel_reference"
  assert load_rl_cfg(TASK).experiment_name == "pikachu_v034/crawl_transfer"


def test_crawl_transfer_matrices_are_finite_and_inverse_on_driven_joints():
  action_map, observation_map = crawl_coordinate_matrices()
  assert action_map.shape == (18, 18)
  assert observation_map.shape == (22, 22)
  assert np.isfinite(action_map).all() and np.isfinite(observation_map).all()
  driven = [
    POLICY_OBSERVATION_JOINT_NAMES.index(name) for name in POLICY_ACTION_JOINT_NAMES
  ]
  np.testing.assert_allclose(
    observation_map[np.ix_(driven, driven)] @ action_map,
    np.eye(len(POLICY_ACTION_JOINT_NAMES)),
    atol=1e-10,
  )


def test_crawl_transfer_has_leg_coupling_and_stable_arm_axis_signs():
  action_map, _ = crawl_coordinate_matrices()
  for side in ("right", "left"):
    leg = [
      POLICY_ACTION_JOINT_NAMES.index(f"{side}_{kind}_joint")
      for kind in _LEG_KINDS
    ]
    leg_map = action_map[np.ix_(leg, leg)]
    assert np.linalg.cond(leg_map) < 4
    assert np.count_nonzero(np.abs(leg_map) > 0.05) > len(_LEG_KINDS)
    arm = [
      POLICY_ACTION_JOINT_NAMES.index(f"{side}_{kind}_joint")
      for kind in _ARM_KINDS
    ]
    np.testing.assert_array_equal(
      action_map[np.ix_(arm, arm)], np.diag([1.0, 1.0, -1.0, -1.0])
    )
