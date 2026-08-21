"""Zero-dimension arm-sine ActionTerm for the arm-ablation curriculum.

``ArmSineAction`` is a **side-effect-only** action term: it contributes
``action_dim == 0`` to the policy's action space, so it never takes space away
from the 10-dim leg ``joint_pos`` term. Instead ``apply_actions()`` — which the
action manager calls on **every decimation physics substep** — writes a sine
target to the four arm joints:

``target = default_arm + amp * sin(2π·freq·t + side_phase)``

where ``side_phase`` is ``0`` for the left arm and ``right_phase_shift`` (π) for
the right, creating an asymmetric disturbance. With ``amp == 0`` the arm is held
at its default pose — exactly the "no swing" baseline.

Curriculum: during training the amplitude ramps from 0 (first ``off_steps``
policy steps learn stable walking with arms still) up to ``default_amp_pitch`` /
``default_amp_roll`` over ``curriculum_ramp_steps``, then stays. The viewer can
override the per-clip amplitude and frequency at play time via the
``amp_override_pitch`` / ``amp_override_roll`` / ``freq_hz`` attributes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.robots.pikachu.pikachu_constants_base import (
  ARM_PITCH_JOINT_NAMES,
  ARM_RIGHT_PHASE_SHIFT,
  ARM_ROLL_JOINT_NAMES,
)
from mjlab.entity import Entity
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["ArmSineAction", "ArmSineActionCfg"]


@dataclass(kw_only=True)
class ArmSineActionCfg(ActionTermCfg):
  """Configuration for the arm-sine side-effect action term."""

  default_amp_pitch: float = 0.6
  """Pitch swing amplitude (rad) once the curriculum saturates."""

  default_amp_roll: float = 0.5
  """Roll swing amplitude (rad) once the curriculum saturates."""

  freq_hz: float = 1.0
  """Swing frequency in Hz (shared by pitch and roll, same phase base)."""

  right_phase_shift: float = ARM_RIGHT_PHASE_SHIFT
  """Phase offset added to the right arm (rad) to drive it anti-phase."""

  curriculum_steps: tuple[int, int] = (24000, 60000)
  """(off_steps, ramp_steps) in policy steps. First ``off_steps`` have amp=0;
   then amplitude ramps linearly over ``ramp_steps`` to the default value."""

  use_curriculum: bool = True
  """Enable the training curriculum. Disabled at play (amp stays 0 until
   overridden by the viewer)."""

  def build(self, env: ManagerBasedRlEnv) -> ArmSineAction:
    return ArmSineAction(self, env)


class ArmSineAction(ActionTerm):
  """Apply a sinusoidal arm swing as an external disturbance."""

  cfg: ArmSineActionCfg
  _entity: Entity

  def __init__(self, cfg: ArmSineActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._angle_ids, self._angle_names = self._entity.find_joints(
      ARM_PITCH_JOINT_NAMES + ARM_ROLL_JOINT_NAMES, preserve_order=True
    )
    self._ids = torch.tensor(self._angle_ids, device=self.device, dtype=torch.long)
    self._default_arm = self._entity.data.default_joint_pos[:, self._ids].clone()
    # Viper-resolved col offsets for pitch- and roll-duty overrides (left/right).
    self._pitch_ids = torch.tensor(
      [self._angle_names.index(n) for n in ARM_PITCH_JOINT_NAMES],
      device=self.device,
      dtype=torch.long,
    )
    self._roll_ids = torch.tensor(
      [self._angle_names.index(n) for n in ARM_ROLL_JOINT_NAMES],
      device=self.device,
      dtype=torch.long,
    )
    self._raw_actions = torch.empty(
      self.num_envs, 0, device=self.device, dtype=torch.float32
    )  # action_dim == 0.
    self._phase = 0.0
    self._off_steps, self._ramp_steps = self.cfg.curriculum_steps
    # play-time overrides (None → use curriculum / default amp).
    self.amp_override_pitch: float | None = None
    self.amp_override_roll: float | None = None
    self.freq_hz: float = self.cfg.freq_hz

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    del actions  # No policy dimensions to consume.

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids  # Phase is global; a single re-zero is fine for ablation reads.
    self._phase = 0.0

  def apply_actions(self) -> None:
    """Advance the swing phase and write the arm target on each substep."""
    self._phase += 2.0 * math.pi * self.freq_hz * self._env.physics_dt
    amp_pitch = self._effective_amp_pitch()
    amp_roll = self._effective_amp_roll()
    if amp_pitch == 0.0 and amp_roll == 0.0:
      target = self._default_arm.clone()
    else:
      sin = math.sin(self._phase)
      right = math.sin(self._phase + self.cfg.right_phase_shift)
      target = self._default_arm.clone()
      pitch = torch.tensor([amp_pitch * sin, amp_pitch * right], device=self.device)
      roll = torch.tensor([amp_roll * sin, amp_roll * right], device=self.device)
      target[:, self._pitch_ids] += pitch
      target[:, self._roll_ids] += roll
    encoder_bias = self._entity.data.encoder_bias[:, self._ids]
    self._entity.set_joint_position_target(target - encoder_bias, joint_ids=self._ids)

  def _effective_amp_pitch(self) -> float:
    if self.amp_override_pitch is not None:
      return self.amp_override_pitch
    return self._curriculum_factor() * self.cfg.default_amp_pitch

  def _effective_amp_roll(self) -> float:
    if self.amp_override_roll is not None:
      return self.amp_override_roll
    return self._curriculum_factor() * self.cfg.default_amp_roll

  def _curriculum_factor(self) -> float:
    """Return 0..1 ramp factor from the training curriculum (or 0 when off)."""
    if not self.cfg.use_curriculum:
      return 0.0
    steps = self._env.common_step_counter
    if steps <= self._off_steps:
      return 0.0
    return min(1.0, (steps - self._off_steps) / self._ramp_steps)

  def set_amps(self, amp_pitch: float, amp_roll: float) -> None:
    """Set play-time override amplitudes (rad) directly from the viewer."""
    self.amp_override_pitch = amp_pitch
    self.amp_override_roll = amp_roll
