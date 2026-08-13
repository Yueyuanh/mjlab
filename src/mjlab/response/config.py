"""Configuration for automated actuator response experiments."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ResponseConfig:
  """Command-line configuration for a suspended response run.

  Each joint is reset to the task's nominal pose before a case. Step tests run
  in both available directions. Sine tests use a symmetric amplitude that is
  derived from the joint's usable range, so commands never cross the configured
  position-limit margin.
  """

  mode: Literal["all", "step", "sine"] = "all"
  """Response family to run."""

  joint_pattern: str = ".*"
  """Regular expression selecting actuated hinge joints."""

  device: str = "cpu"
  """MJWarp device. CPU is efficient and deterministic for one suspended robot."""

  output_root: str = "logs/response"
  """Root directory for timestamped CSV and JSON result folders."""

  suspend_height_m: float = 1.0
  """Fixed root height. No terrain is present regardless of this value."""

  limit_margin_fraction: float = 0.1
  """Fraction removed from each end of every finite joint range."""

  step_amplitude_fraction: float = 0.25
  """Fraction of the available one-sided travel used by each step."""

  step_pre_s: float = 0.4
  """Nominal-pose settling time recorded before a step."""

  step_duration_s: float = 1.6
  """Recorded time after a step."""

  sine_frequencies_hz: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
  """Discrete sine-response frequencies."""

  sine_amplitude_fraction: float = 0.2
  """Fraction of symmetric usable travel used by sine commands."""

  sine_pre_s: float = 0.4
  """Nominal-pose settling time before each sinusoid."""

  sine_cycles: float = 4.0
  """Number of recorded cycles at each sine frequency."""

  sine_warmup_cycles: float = 1.0
  """Initial sine cycles excluded from frequency-domain fitting."""

  sample_stride: int = 1
  """Write one CSV sample every N physics steps."""

  strict_no_contact: bool = True
  """Abort if any contact appears despite the contact-free scene construction."""

  torque_safety_factor: float = 1.5
  """Multiplier used for the motor-sizing recommendation in the summary."""

  speed_safety_factor: float = 1.2
  """Multiplier used for the speed-sizing recommendation in the summary."""

  def validate(self) -> None:
    """Raise a useful error for unsafe or nonsensical settings."""
    if not 0.0 <= self.limit_margin_fraction < 0.5:
      raise ValueError("limit_margin_fraction must be in [0, 0.5).")
    if not 0.0 < self.step_amplitude_fraction <= 1.0:
      raise ValueError("step_amplitude_fraction must be in (0, 1].")
    if not 0.0 < self.sine_amplitude_fraction <= 1.0:
      raise ValueError("sine_amplitude_fraction must be in (0, 1].")
    if self.step_pre_s < 0.0 or self.step_duration_s <= 0.0:
      raise ValueError("Step durations must be non-negative, with duration > 0.")
    if self.sine_pre_s < 0.0 or self.sine_cycles <= 0.0:
      raise ValueError("Sine pre-time must be non-negative and cycles must be > 0.")
    if self.sine_warmup_cycles < 0.0:
      raise ValueError("sine_warmup_cycles must be non-negative.")
    if not self.sine_frequencies_hz or any(f <= 0.0 for f in self.sine_frequencies_hz):
      raise ValueError("sine_frequencies_hz must contain only positive values.")
    if self.sample_stride < 1:
      raise ValueError("sample_stride must be at least 1.")
    if self.torque_safety_factor < 1.0 or self.speed_safety_factor < 1.0:
      raise ValueError("Motor-sizing safety factors must be at least 1.0.")
