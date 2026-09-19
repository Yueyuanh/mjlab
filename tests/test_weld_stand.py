"""Regression tests for ``play --weld`` (the welded rigid monolith).

``--weld`` replaces every internal joint with an ``mjEQ_WELD`` equality so the
robot becomes a single rigid body (one free-jointed base, no actuators). These
tests keep the build healthy without needing a GPU or a full RL env:
1. the welded model really is a monolith (njnt==1, nu==0, >0 welds);
2. ``_apply_weld_cfg`` empties every joint-driven manager and forces the
   freeze/balance hacks off.
"""

import pytest

from mjlab.scene import Scene
from mjlab.scripts.play import PlayConfig, _apply_weld_cfg
from mjlab.tasks.registry import list_tasks, load_env_cfg


def _pikachu_27dof_task_id() -> str:
  """Return the Pikachu 27-dof velocity task, skipping if unregistered."""
  candidates = [
    t for t in list_tasks() if "Velocity" in t and "Pikachu" in t and "27dof" in t
  ]
  if not candidates:
    pytest.skip("no Pikachu 27-dof velocity task in the task pool")
  return candidates[0]


def test_weld_model_is_rigid_monolith() -> None:
  """Compiling the welded scene yields one 6-dof base and zero actuators."""
  task_id = _pikachu_27dof_task_id()
  cfg = load_env_cfg(task_id, play=True)
  _apply_weld_cfg(cfg, PlayConfig(agent="frozen", weld=True))
  model = Scene(cfg.scene, device="cpu").compile()
  assert model.njnt == 1  # only the root free joint survives
  assert model.nu == 0  # every actuator is gone
  assert model.nv == 6  # no internal hinge/ball dofs remain
  assert model.neq > 0  # the internal links are welded together


def test_weld_cfg_empties_joint_driven_managers() -> None:
  """``_apply_weld_cfg`` neutralizes joint-driven managers/hacks contract-wise."""
  task_id = _pikachu_27dof_task_id()
  cfg = load_env_cfg(task_id, play=True)
  out = _apply_weld_cfg(cfg, PlayConfig(agent="frozen", weld=True, balance=True))
  # Managers that touch joints must be empty so nothing writes into these slices.
  assert cfg.actions == {}
  assert cfg.observations == {}
  assert cfg.rewards == {}
  assert cfg.terminations == {}
  assert cfg.metrics == {}
  assert cfg.events.keys() == {"reset_base"}
  # The freeze/balance hacks are meaningless on a monolith and get forced off.
  assert out.weld is True
  assert out.freeze_joints is False
  assert out.balance is False


def test_weld_requires_dummy_agent() -> None:
  """``--weld`` with a trained agent should not silently build a broken env."""
  task_id = _pikachu_27dof_task_id()
  cfg = load_env_cfg(task_id, play=True)
  with pytest.raises(ValueError) as exc:
    _apply_weld_cfg(cfg, PlayConfig(agent="trained", weld=True))
  assert "dummy" in str(exc.value)
