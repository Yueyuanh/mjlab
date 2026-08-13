"""Limit-aware command generation for step and sine response tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from mjlab.response.config import ResponseConfig


@dataclass(frozen=True)
class ResponseCase:
  """One independently reset response experiment."""

  case_id: str
  response_type: Literal["step", "sine"]
  joint: str
  center_rad: float
  amplitude_rad: float
  pre_duration_s: float
  stimulus_duration_s: float
  direction: int = 0
  frequency_hz: float = 0.0
  warmup_cycles: float = 0.0

  @property
  def duration_s(self) -> float:
    return self.pre_duration_s + self.stimulus_duration_s

  def command(self, test_time_s: float) -> tuple[float, float]:
    """Return position and nominal velocity commands at a case-local time."""
    stimulus_time = test_time_s - self.pre_duration_s
    if stimulus_time < 0.0:
      return self.center_rad, 0.0
    if self.response_type == "step":
      return self.center_rad + self.amplitude_rad, 0.0
    omega = 2.0 * np.pi * self.frequency_hz
    phase = omega * stimulus_time
    return (
      self.center_rad + self.amplitude_rad * float(np.sin(phase)),
      self.amplitude_rad * omega * float(np.cos(phase)),
    )


def usable_limits(
  lower: float,
  upper: float,
  margin_fraction: float,
) -> tuple[float, float]:
  """Shrink finite hard limits by a symmetric fractional safety margin."""
  if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
    raise ValueError("Response tests require finite, ordered joint limits.")
  margin = (upper - lower) * margin_fraction
  return lower + margin, upper - margin


def build_joint_cases(
  joint: str,
  home_rad: float,
  lower_rad: float,
  upper_rad: float,
  cfg: ResponseConfig,
) -> list[ResponseCase]:
  """Create limit-safe positive/negative steps and discrete sine cases."""
  safe_lower, safe_upper = usable_limits(
    lower_rad, upper_rad, cfg.limit_margin_fraction
  )
  center = float(np.clip(home_rad, safe_lower, safe_upper))
  positive_room = safe_upper - center
  negative_room = center - safe_lower
  cases: list[ResponseCase] = []

  if cfg.mode in ("all", "step"):
    for direction, room, label in (
      (1, positive_room, "positive"),
      (-1, negative_room, "negative"),
    ):
      amplitude = direction * room * cfg.step_amplitude_fraction
      if abs(amplitude) <= 1e-9:
        continue
      cases.append(
        ResponseCase(
          case_id=f"{joint}__step_{label}",
          response_type="step",
          joint=joint,
          center_rad=center,
          amplitude_rad=amplitude,
          pre_duration_s=cfg.step_pre_s,
          stimulus_duration_s=cfg.step_duration_s,
          direction=direction,
        )
      )

  if cfg.mode in ("all", "sine"):
    symmetric_room = min(positive_room, negative_room)
    amplitude = symmetric_room * cfg.sine_amplitude_fraction
    if amplitude > 1e-9:
      for frequency in cfg.sine_frequencies_hz:
        frequency_label = f"{frequency:g}".replace(".", "p")
        cases.append(
          ResponseCase(
            case_id=f"{joint}__sine_{frequency_label}hz",
            response_type="sine",
            joint=joint,
            center_rad=center,
            amplitude_rad=amplitude,
            pre_duration_s=cfg.sine_pre_s,
            stimulus_duration_s=cfg.sine_cycles / frequency,
            frequency_hz=frequency,
            warmup_cycles=cfg.sine_warmup_cycles,
          )
        )
  return cases
