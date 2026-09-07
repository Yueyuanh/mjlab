"""Zero-dimension random-flail ActionTerm for the upper-limb disturbance.

``RandomFlailAction`` is a **side-effect-only** action term: it contributes
``action_dim == 0`` to the policy's action space (so it never eats into the
10-dim leg ``joint_pos`` term). On every decimation physics substep
``apply_actions()`` writes a **piecewise-constant random target** to the given
upper-limb joints:

  - Every env keeps a per-env ``_timer``; when it hits 0 a fresh offset is
    sampled ``uniform(-1, 1)`` per joint and scaled to ``flail_ratio ×
    semi_range`` around each joint's soft-limit centre, then ``_timer`` is
    re-rolled from ``refresh_interval_range``.
  - Between refreshes the offset (and hence the PD target) is held constant,
    so the arm/ear/tail holds random poses and *snaps* to a new one after each
    refresh — a much richer, unpredictable disturbance than a fixed sine.

With curriculum factor ``0`` the offset is zeroed and the limbs are held at
their default pose — exactly the "no flail" baseline. During training the
amplitude ramps 0 → ``flail_ratio``, letting leg-only walking learn first
before the upper limbs start moving.

The policy observes all 27 joint positions/velocities, so it can sense and
compensate the limb motion; the action never controls those joints directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["RandomFlailAction", "RandomFlailActionCfg"]


@dataclass(kw_only=True)
class RandomFlailActionCfg(ActionTermCfg):
  """Configuration for the random-flail side-effect action term."""

  joint_names: tuple[str, ...]
  """Upper-limb joints to flail (e.g. ``ARM_JOINT_NAMES`` or
  ``EAR_JOINT_NAMES + TAIL_JOINT_NAMES``)."""

  flail_ratio: float = 0.6
  """Max flail amplitude as a fraction of each joint's soft-limit semi-range."""

  refresh_interval_range: tuple[float, float] = (0.4, 1.0)
  """Per-environment random hold duration before re-sampling the pose (s)."""

  curriculum_steps: tuple[int, int] = (24000, 60000)
  """(off_steps, ramp_steps) in policy steps. First ``off_steps`` the limbs are
  held; amplitude then ramps linearly to ``flail_ratio``."""

  use_curriculum: bool = True
  """Enable the training curriculum. Disabled at play (limbs stay still until a
  viewer override sets ``flail_ratio``)."""

  def build(self, env: ManagerBasedRlEnv) -> RandomFlailAction:
    return RandomFlailAction(self, env)


class RandomFlailAction(ActionTerm):
  """Apply random upper-limb pose snapping as an external disturbance."""

  cfg: RandomFlailActionCfg
  _entity: Entity
  _ids: torch.Tensor
  _center: torch.Tensor  # (num_envs, n_joint) default pose.
  _half_width: torch.Tensor  # (num_envs, n_joint) soft-limit semi-range.
  _lo: torch.Tensor
  _hi: torch.Tensor
  _offset: torch.Tensor  # (num_envs, n_joint) current uniform(-1,1) offset.
  _timer: torch.Tensor  # (num_envs,) seconds until next refresh.

  def __init__(self, cfg: RandomFlailActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    joint_ids, _ = self._entity.find_joints(
      self.cfg.joint_names, preserve_order=True
    )
    self._ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._raw_actions = torch.empty(
      self.num_envs, 0, device=self.device, dtype=torch.float32
    )

    limits = self._entity.data.soft_joint_pos_limits[:, self._ids].clone()
    self._lo = limits[..., 0]
    self._hi = limits[..., 1]
    self._half_width = 0.5 * (self._hi - self._lo)
    self._center = self._entity.data.default_joint_pos[:, self._ids].clone()

    self._offset = torch.zeros(
      self.num_envs, len(cfg.joint_names), device=self.device
    )
    self._timer = torch.zeros(self.num_envs, device=self.device)
    # Prime per-env random pose/timer so the very first substep moves limbs.
    self.reset()

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    del actions  # No policy dimensions to consume.

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    if isinstance(env_ids, slice):
      n = self.num_envs
    else:
      n = int(env_ids.numel())
    shapes = (n, len(self.cfg.joint_names))
    new_off = torch.empty(shapes, device=self.device).uniform_(-1.0, 1.0)
    lo, hi = self.cfg.refresh_interval_range
    new_timer = torch.empty((n,), device=self.device).uniform_(lo, hi)
    self._offset[env_ids] = new_off
    self._timer[env_ids] = new_timer

  def apply_actions(self) -> None:
    """Refresh elapsed limbs and write their piecewise-constant targets."""
    self._timer -= self._env.physics_dt
    refresh = self._timer <= 0.0
    n_refresh = int(refresh.sum().item())
    if n_refresh:
      refreshed = torch.nonzero(refresh).flatten()
      self.reset(refreshed)
    self._write_target()

  def _write_target(self) -> None:
    factor = self._curriculum_factor() if self.cfg.use_curriculum else 1.0
    scale = self.cfg.flail_ratio * self._half_width
    target = self._center + self._offset * scale * factor
    target = torch.clamp(target, self._lo, self._hi)
    encoder_bias = self._entity.data.encoder_bias[:, self._ids]
    self._entity.set_joint_position_target(
      target - encoder_bias, joint_ids=self._ids
    )

  def _curriculum_factor(self) -> float:
    """Return 0..1 ramp factor from the training curriculum (or 0 when off)."""
    if not self.cfg.use_curriculum:
      return 0.0
    steps = self._env.common_step_counter
    off_steps, ramp_steps = self.cfg.curriculum_steps
    if steps <= off_steps:
      return 0.0
    return min(1.0, (steps - off_steps) / ramp_steps)