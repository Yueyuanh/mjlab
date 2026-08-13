"""Time- and frequency-domain analysis for actuator response CSV rows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from mjlab.response.config import ResponseConfig
from mjlab.response.model import JointModel, model_predictions

Row = Mapping[str, Any]


def _values(rows: list[Row], key: str) -> np.ndarray:
  return np.asarray([float(row[key]) for row in rows], dtype=float)


def _first_crossing(
  time: np.ndarray,
  value: np.ndarray,
  threshold: float,
) -> float:
  indexes = np.flatnonzero(value >= threshold)
  if not len(indexes):
    return float("nan")
  index = int(indexes[0])
  if index == 0:
    return float(time[0])
  t0, t1 = time[index - 1], time[index]
  y0, y1 = value[index - 1], value[index]
  if y1 == y0:
    return float(t1)
  fraction = np.clip((threshold - y0) / (y1 - y0), 0.0, 1.0)
  return float(t0 + fraction * (t1 - t0))


def _settling_time(
  time: np.ndarray,
  value: np.ndarray,
  target: float,
  band: float,
) -> float:
  outside = np.flatnonzero(np.abs(value - target) > band)
  if not len(outside):
    return 0.0
  candidate = int(outside[-1]) + 1
  if candidate >= len(time):
    return float("nan")
  return float(time[candidate])


def _base_summary(rows: list[Row]) -> dict[str, Any]:
  first = rows[0]
  return {
    "record_type": str(first["response_type"]),
    "case_id": str(first["case_id"]),
    "joint": str(first["joint"]),
    "motor_model": str(first["motor_model"]),
    "frequency_hz": float(first["frequency_hz"]),
    "direction": int(first["direction"]),
    "amplitude_rad": float(first["amplitude_rad"]),
    "samples": len(rows),
  }


def analyze_step_case(rows: list[Row]) -> dict[str, Any]:
  """Compute conventional 5/10/50/90%, overshoot, and settling metrics."""
  result = _base_summary(rows)
  time = _values(rows, "test_time_s")
  command = _values(rows, "command_pos_rad")
  measured = _values(rows, "measured_pos_rad")
  velocity = _values(rows, "measured_vel_rad_s")
  torque = _values(rows, "actuator_torque_nm")
  stimulus_start = float(rows[0]["stimulus_start_s"])
  pre = time < stimulus_start
  post = time >= stimulus_start
  if not np.any(post):
    raise ValueError(f"Step case {result['case_id']} has no post-step samples.")

  baseline = float(
    np.median(measured[pre]) if np.any(pre) else measured[np.flatnonzero(post)[0]]
  )
  target = float(np.median(command[post]))
  amplitude = target - baseline
  magnitude = abs(amplitude)
  if magnitude <= 1e-12:
    raise ValueError(f"Step case {result['case_id']} has zero command amplitude.")

  sign = 1.0 if amplitude > 0.0 else -1.0
  post_time = time[post] - stimulus_start
  directed = sign * (measured[post] - baseline)
  latency_5 = _first_crossing(post_time, directed, 0.05 * magnitude)
  t10 = _first_crossing(post_time, directed, 0.10 * magnitude)
  t50 = _first_crossing(post_time, directed, 0.50 * magnitude)
  t90 = _first_crossing(post_time, directed, 0.90 * magnitude)
  final_count = max(3, int(np.count_nonzero(post) * 0.1))
  final_value = float(np.median(measured[post][-final_count:]))
  error = command[post] - measured[post]
  peak_index = int(np.argmax(directed))
  overshoot_fraction = max(0.0, directed[peak_index] / magnitude - 1.0)
  damping_ratio = float("nan")
  if overshoot_fraction > 1e-9:
    log_mp = np.log(overshoot_fraction)
    damping_ratio = float(-log_mp / np.sqrt(np.pi**2 + log_mp**2))
  rise_time = t90 - t10 if np.isfinite(t10) and np.isfinite(t90) else float("nan")

  result.update(
    {
      "baseline_rad": baseline,
      "target_rad": target,
      "final_value_rad": final_value,
      "steady_state_error_rad": target - final_value,
      "steady_state_error_pct": 100.0 * abs(target - final_value) / magnitude,
      "latency_5pct_s": latency_5,
      "delay_50pct_s": t50,
      "rise_time_10_90_s": rise_time,
      "peak_time_s": float(post_time[peak_index]),
      "overshoot_pct": 100.0 * overshoot_fraction,
      "settling_time_2pct_s": _settling_time(
        post_time, measured[post], target, 0.02 * magnitude
      ),
      "settling_time_5pct_s": _settling_time(
        post_time, measured[post], target, 0.05 * magnitude
      ),
      "rmse_rad": float(np.sqrt(np.mean(error**2))),
      "iae_rad_s": float(np.trapezoid(np.abs(error), post_time)),
      "ise_rad2_s": float(np.trapezoid(error**2, post_time)),
      "peak_velocity_rad_s": float(np.max(np.abs(velocity[post]))),
      "peak_torque_nm": float(np.max(np.abs(torque[post]))),
      "torque_rms_nm": float(np.sqrt(np.mean(torque[post] ** 2))),
      "saturation_fraction": float(
        np.mean(_values(rows, "torque_saturated")[post] > 0.5)
      ),
      "observed_damping_ratio": damping_ratio,
      "rise_time_bandwidth_hz": (
        0.35 / rise_time if np.isfinite(rise_time) and rise_time > 0.0 else np.nan
      ),
    }
  )
  return result


def _fit_sine(
  time: np.ndarray,
  value: np.ndarray,
  frequency_hz: float,
) -> tuple[float, float, float, float]:
  omega_t = 2.0 * np.pi * frequency_hz * time
  design = np.column_stack((np.sin(omega_t), np.cos(omega_t), np.ones_like(time)))
  coeff, _, _, _ = np.linalg.lstsq(design, value, rcond=None)
  fitted = design @ coeff
  residual = value - fitted
  total = value - np.mean(value)
  denominator = float(np.dot(total, total))
  r_squared = (
    1.0 - float(np.dot(residual, residual)) / denominator
    if denominator > 1e-18
    else float("nan")
  )
  amplitude = float(np.hypot(coeff[0], coeff[1]))
  phase = float(np.arctan2(coeff[1], coeff[0]))
  return amplitude, phase, float(coeff[2]), r_squared


def _harmonic_distortion(
  time: np.ndarray,
  value: np.ndarray,
  frequency_hz: float,
  max_harmonic: int = 5,
) -> float:
  """Fit harmonics jointly so partial cycles cannot leak the fundamental."""
  omega_t = 2.0 * np.pi * frequency_hz * time
  columns = []
  for harmonic in range(1, max_harmonic + 1):
    columns.extend((np.sin(harmonic * omega_t), np.cos(harmonic * omega_t)))
  design = np.column_stack((*columns, np.ones_like(time)))
  coeff, _, _, _ = np.linalg.lstsq(design, value, rcond=None)
  amplitudes = np.hypot(coeff[0:-1:2], coeff[1:-1:2])
  if amplitudes[0] <= 1e-12:
    return float("nan")
  return float(np.sqrt(np.sum(amplitudes[1:] ** 2)) / amplitudes[0])


def _wrap_phase_degrees(value: float) -> float:
  return (value + 180.0) % 360.0 - 180.0


def analyze_sine_case(rows: list[Row]) -> dict[str, Any]:
  """Fit the fundamental response and report Bode, distortion, and demand data."""
  result = _base_summary(rows)
  frequency = float(result["frequency_hz"])
  time = _values(rows, "test_time_s")
  stimulus_start = float(rows[0]["stimulus_start_s"])
  warmup_cycles = float(rows[0]["warmup_cycles"])
  analysis_start = stimulus_start + warmup_cycles / frequency
  keep = time >= analysis_start
  if np.count_nonzero(keep) < 8:
    raise ValueError(f"Sine case {result['case_id']} has too few fitted samples.")
  fit_time = time[keep] - stimulus_start
  command = _values(rows, "command_pos_rad")[keep]
  measured = _values(rows, "measured_pos_rad")[keep]
  velocity = _values(rows, "measured_vel_rad_s")[keep]
  torque = _values(rows, "actuator_torque_nm")[keep]
  power = _values(rows, "mechanical_power_w")[keep]

  command_amp, command_phase, _, command_r2 = _fit_sine(fit_time, command, frequency)
  output_amp, output_phase, output_offset, output_r2 = _fit_sine(
    fit_time, measured, frequency
  )
  gain = output_amp / command_amp if command_amp > 1e-12 else float("nan")
  phase_deg = _wrap_phase_degrees(np.degrees(output_phase - command_phase))
  thd = _harmonic_distortion(fit_time, measured, frequency)

  result.update(
    {
      "analysis_start_s": analysis_start,
      "command_amplitude_fit_rad": command_amp,
      "response_amplitude_rad": output_amp,
      "response_offset_rad": output_offset,
      "gain_ratio": gain,
      "gain_db": 20.0 * np.log10(gain) if gain > 0.0 else float("-inf"),
      "phase_deg": phase_deg,
      "equivalent_delay_s": -np.radians(phase_deg) / (2.0 * np.pi * frequency),
      "command_fit_r2": command_r2,
      "response_fit_r2": output_r2,
      "thd_pct": 100.0 * thd,
      "rmse_rad": float(np.sqrt(np.mean((command - measured) ** 2))),
      "peak_velocity_rad_s": float(np.max(np.abs(velocity))),
      "peak_torque_nm": float(np.max(np.abs(torque))),
      "torque_rms_nm": float(np.sqrt(np.mean(torque**2))),
      "peak_mechanical_power_w": float(np.max(np.abs(power))),
      "saturation_fraction": float(
        np.mean(_values(rows, "torque_saturated")[keep] > 0.5)
      ),
    }
  )
  return result


def _bandwidth_hz(sine_rows: list[dict[str, Any]]) -> float:
  valid = sorted(
    (
      (float(row["frequency_hz"]), float(row["gain_ratio"]))
      for row in sine_rows
      if np.isfinite(float(row.get("gain_ratio", np.nan)))
      and float(row["gain_ratio"]) > 0.0
    ),
    key=lambda item: item[0],
  )
  if len(valid) < 2:
    return float("nan")
  reference = valid[0][1]
  threshold = reference / np.sqrt(2.0)
  for index in range(1, len(valid)):
    f0, g0 = valid[index - 1]
    f1, g1 = valid[index]
    if g1 > threshold:
      continue
    if g1 == g0:
      return f1
    fraction = (threshold - g0) / (g1 - g0)
    return float(np.exp(np.log(f0) + fraction * (np.log(f1) - np.log(f0))))
  return float("nan")


def _joint_demand_summary(
  joint: str,
  rows: list[Row],
  case_summaries: list[dict[str, Any]],
  model: JointModel,
  cfg: ResponseConfig,
) -> dict[str, Any]:
  torque = _values(rows, "actuator_torque_nm")
  velocity = _values(rows, "measured_vel_rad_s")
  power = _values(rows, "mechanical_power_w")
  peak_torque = float(np.max(np.abs(torque)))
  rms_torque = float(np.sqrt(np.mean(torque**2)))
  peak_speed = float(np.max(np.abs(velocity)))
  peak_power = float(np.max(np.abs(power)))
  sine = [row for row in case_summaries if row["record_type"] == "sine"]
  step = [row for row in case_summaries if row["record_type"] == "step"]
  torque_limit = model.torque_limit_nm
  speed_limit = model.velocity_limit_rad_s
  return {
    "record_type": "joint_demand",
    "case_id": f"{joint}__aggregate",
    "joint": joint,
    "motor_model": model.motor_model,
    "samples": len(rows),
    "peak_torque_nm": peak_torque,
    "torque_rms_nm": rms_torque,
    "peak_velocity_rad_s": peak_speed,
    "peak_mechanical_power_w": peak_power,
    "absolute_energy_j": float(
      sum(float(row["absolute_energy_j_increment"]) for row in rows)
    ),
    "recommended_peak_torque_nm": peak_torque * cfg.torque_safety_factor,
    "recommended_continuous_torque_nm": rms_torque * cfg.torque_safety_factor,
    "recommended_speed_rad_s": peak_speed * cfg.speed_safety_factor,
    "torque_headroom_pct": (
      100.0 * (torque_limit - peak_torque) / torque_limit
      if np.isfinite(torque_limit) and torque_limit > 0.0
      else np.nan
    ),
    "speed_headroom_pct": (
      100.0 * (speed_limit - peak_speed) / speed_limit
      if np.isfinite(speed_limit) and speed_limit > 0.0
      else np.nan
    ),
    "measured_bandwidth_hz": _bandwidth_hz(sine),
    "worst_step_overshoot_pct": (
      max(float(row["overshoot_pct"]) for row in step) if step else np.nan
    ),
    "worst_step_settling_2pct_s": (
      max(float(row["settling_time_2pct_s"]) for row in step) if step else np.nan
    ),
    "max_torque_saturation_fraction": max(
      (float(row["saturation_fraction"]) for row in case_summaries),
      default=0.0,
    ),
    **model_predictions(model),
    **{
      key: value
      for key, value in model.as_dict().items()
      if key not in {"joint", "motor_model"}
    },
  }


def analyze_response_rows(
  rows: Iterable[Row],
  joint_models: Mapping[str, JointModel],
  cfg: ResponseConfig,
) -> list[dict[str, Any]]:
  """Analyze all cases, then append one motor-demand row per tested joint."""
  by_case: dict[str, list[Row]] = defaultdict(list)
  by_joint: dict[str, list[Row]] = defaultdict(list)
  all_rows = list(rows)
  for row in all_rows:
    by_case[str(row["case_id"])].append(row)
    by_joint[str(row["joint"])].append(row)

  summaries: list[dict[str, Any]] = []
  by_joint_summaries: dict[str, list[dict[str, Any]]] = defaultdict(list)
  for case_rows in by_case.values():
    case_rows.sort(key=lambda row: float(row["test_time_s"]))
    response_type = str(case_rows[0]["response_type"])
    summary = (
      analyze_step_case(case_rows)
      if response_type == "step"
      else analyze_sine_case(case_rows)
    )
    model = joint_models[str(summary["joint"])]
    summary.update(model_predictions(model))
    summaries.append(summary)
    by_joint_summaries[str(summary["joint"])].append(summary)

  for joint, joint_rows in by_joint.items():
    summaries.append(
      _joint_demand_summary(
        joint,
        joint_rows,
        by_joint_summaries[joint],
        joint_models[joint],
        cfg,
      )
    )
  return summaries
