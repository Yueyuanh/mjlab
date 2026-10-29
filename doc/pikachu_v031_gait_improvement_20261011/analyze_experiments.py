"""Analyze E0--E4 training logs and fixed-command rollout CSVs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util import tensor_util

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PLOTS = OUT / "plots"
METRICS = OUT / "metrics"
RESULTS = OUT / "manifests" / "run_results.json"
JOINT_LIMIT = 0.26

TRAIN_TAGS = (
  "Train/mean_reward",
  "Train/mean_episode_length",
  "Policy/mean_std",
  "Loss/symmetry",
  "Episode_Metrics/vel/err_yaw",
  "Episode_Metrics/joint_position_rad/left_hip_roll_joint",
  "Episode_Metrics/joint_position_rad/right_hip_roll_joint",
)


def _scalar(value) -> float | None:
  if value.HasField("tensor"):
    array = tensor_util.make_ndarray(value.tensor)
    return float(np.asarray(array).reshape(-1)[0]) if array.size else None
  if value.HasField("simple_value"):
    return float(value.simple_value)
  return None


def load_training(event_file: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
  values: dict[str, list[tuple[int, float]]] = {tag: [] for tag in TRAIN_TAGS}
  for event in EventFileLoader(str(event_file)).Load():
    if not event.HasField("summary"):
      continue
    for item in event.summary.value:
      if item.tag not in values:
        continue
      scalar = _scalar(item)
      if scalar is not None and np.isfinite(scalar):
        values[item.tag].append((event.step, scalar))
  return {
    tag: (
      np.asarray([x[0] for x in points], dtype=float),
      np.asarray([x[1] for x in points], dtype=float),
    )
    for tag, points in values.items()
    if points
  }


def _csv_data(path: Path) -> dict[str, np.ndarray]:
  with path.open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
  result: dict[str, np.ndarray] = {}
  for key in rows[0]:
    try:
      result[key] = np.asarray([float(row[key]) for row in rows], dtype=float)
    except (TypeError, ValueError):
      pass
  return result


def _mean(x: np.ndarray) -> float:
  return float(np.mean(x)) if x.size else float("nan")


def _rms(x: np.ndarray) -> float:
  return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")


def summarize_rollout(path: Path, expected_s: float) -> dict[str, float | bool | str]:
  d = _csv_data(path)
  t = d["time_s"]
  mask = t >= min(0.5, float(t[-1]))
  lp = d["joint_position_rad/left_hip_roll_joint"][mask]
  rp = d["joint_position_rad/right_hip_roll_joint"][mask]
  lt = d["joint_target_rad/left_hip_roll_joint"][mask]
  rt = d["joint_target_rad/right_hip_roll_joint"][mask]
  yaw = np.unwrap(d["vel/yaw_w"])
  l_span = float(np.ptp(lp))
  r_span = float(np.ptp(rp))
  return {
    "file": str(path.relative_to(ROOT)),
    "duration_s": float(t[-1] + (t[1] - t[0] if len(t) > 1 else 0.0)),
    "survived": bool(float(t[-1]) >= expected_s - 0.05),
    "act_x_mean_m_s": _mean(d["vel/act_x"][mask]),
    "act_y_rms_m_s": _rms(d["vel/act_y"][mask]),
    "yaw_rms_rad_s": _rms(d["vel/act_yaw"][mask]),
    "heading_change_deg": float(np.degrees(yaw[-1] - yaw[0])),
    "left_roll_mean_rad": _mean(lp),
    "right_roll_mean_rad": _mean(rp),
    "roll_signed_mirror_residual_rad": abs(_mean(lp) + _mean(rp)),
    "roll_abs_mean_gap_rad": abs(_mean(np.abs(lp)) - _mean(np.abs(rp))),
    "roll_span_ratio": min(l_span, r_span) / max(l_span, r_span, 1e-12),
    "left_target_abs_max_rad": float(np.max(np.abs(lt))),
    "right_target_abs_max_rad": float(np.max(np.abs(rt))),
    "target_over_hard_limit_fraction": float(
      np.mean(np.concatenate((np.abs(lt), np.abs(rt))) > JOINT_LIMIT)
    ),
    "raw_action_abs_max": float(np.max(d.get("policy/raw_action_abs_max", [np.nan]))),
    "raw_action_clip_fraction": _mean(
      d.get("policy/raw_action_clip_fraction", np.asarray([np.nan]))
    ),
  }


def collect_rollouts() -> dict[str, dict[str, Any]]:
  summaries: dict[str, dict[str, Any]] = {}
  for index_path in sorted(METRICS.rglob("*_index.json")):
    index = json.loads(index_path.read_text(encoding="utf-8"))
    tag = index["tag"]
    for clip in index["clips"]:
      csv_path = index_path.parent / clip["file"]
      summaries[f"{tag}/{clip['clip']}"] = summarize_rollout(
        csv_path, float(index["seconds_per_clip"])
      )
  return summaries


def collect_training(
  results: dict[str, Any],
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
  return {
    exp: load_training(ROOT / info["event_file"])
    for exp, info in results["runs"].items()
  }


def summarize_training(
  all_series: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
) -> dict[str, dict[str, float]]:
  summaries: dict[str, dict[str, float]] = {}
  for exp, series in all_series.items():
    row: dict[str, float] = {}
    for tag in (
      "Train/mean_reward",
      "Train/mean_episode_length",
      "Policy/mean_std",
      "Loss/symmetry",
    ):
      if tag not in series:
        continue
      x, y = series[tag]
      key = tag.replace("/", "_").lower()
      row[f"{key}_last"] = float(y[-1])
      row[f"{key}_best"] = float(np.max(y))
      row[f"{key}_last_iteration"] = float(x[-1])
    summaries[exp] = row
  return summaries


def plot_training(
  results: dict[str, Any],
  all_series: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
) -> None:
  fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
  specs = (
    ("Train/mean_reward", "Mean reward", axes[0, 0]),
    ("Train/mean_episode_length", "Episode length", axes[0, 1]),
    ("Policy/mean_std", "Policy action std", axes[1, 0]),
    ("Loss/symmetry", "Symmetry loss", axes[1, 1]),
  )
  for exp, info in results["runs"].items():
    del info
    series = all_series[exp]
    for tag, title, ax in specs:
      if tag not in series:
        continue
      x, y = series[tag]
      ax.plot(x, y, lw=1.1, label=exp)
      ax.set_title(title)
      ax.set_xlabel("Iteration")
      ax.set_xlim(0, 5000)
      ax.grid(alpha=0.25)
  for ax in axes.flat:
    if ax.lines:
      ax.axvline(1000, color="#999999", ls="--", lw=0.8)
      ax.axvline(2000, color="#999999", ls="-.", lw=0.8)
      ax.axvline(3000, color="#999999", ls=":", lw=0.8)
      ax.legend()
  fig.suptitle("V031 E0-E4 training curves (first 5k iterations)")
  fig.savefig(PLOTS / "training_comparison.png", dpi=180)
  plt.close(fig)


def plot_rollouts(summaries: dict[str, dict[str, Any]]) -> None:
  keys = [f"E{i}_nominal_s42/fwd" for i in range(5)]
  keys = [key for key in keys if key in summaries]
  labels = [key.split("_")[0] for key in keys]
  fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
  metrics = (
    ("act_x_mean_m_s", "Forward speed (target 0.4)", "m/s", False, 0.4),
    ("yaw_rms_rad_s", "Forward yaw RMS", "rad/s", False, None),
    ("heading_change_deg", "8 s |heading change|", "deg", True, None),
    ("roll_abs_mean_gap_rad", "Hip-roll |mean| gap", "rad", False, None),
    ("roll_span_ratio", "Hip-roll min/max span ratio", "ratio", False, 1.0),
    (
      "raw_action_clip_fraction",
      "Raw action clip fraction",
      "fraction",
      False,
      None,
    ),
  )
  for ax, (name, title, unit, absolute, target) in zip(axes.flat, metrics, strict=True):
    values = [float(summaries[key][name]) for key in keys]
    if absolute:
      values = [abs(value) for value in values]
    ax.bar(
      labels,
      values,
      color=["#777777", "#4c78a8", "#59a14f", "#f28e2b", "#b07aa1"][: len(keys)],
    )
    if target is not None:
      ax.axhline(target, color="#d62728", ls="--", lw=1.0)
    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.grid(axis="y", alpha=0.25)
  fig.suptitle("Fixed 0.4 m/s nominal replay")
  fig.savefig(PLOTS / "rollout_comparison.png", dpi=180)
  plt.close(fig)


def plot_final_candidate(summaries: dict[str, dict[str, Any]]) -> None:
  keys = (
    "E0final_nominal_s42/fwd",
    "E4pre_full_nominal_s42/fwd",
    "E4final_nominal_s42/fwd",
  )
  if not all(key in summaries for key in keys):
    return
  labels = ("E0 m29999", "E4 m2500", "E4 m5000")
  colors = ("#777777", "#59a14f", "#b07aa1")
  fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
  specs = (
    ("act_x_mean_m_s", "Forward speed (target 0.4)", "m/s", 0.4),
    ("yaw_rms_rad_s", "Yaw RMS", "rad/s", None),
    ("roll_abs_mean_gap_rad", "Hip-roll |mean| gap", "rad", None),
    ("raw_action_clip_fraction", "Raw action clip fraction", "fraction", None),
  )
  for ax, (metric, title, unit, target) in zip(axes.flat, specs, strict=True):
    values = [float(summaries[key][metric]) for key in keys]
    ax.bar(labels, values, color=colors)
    if target is not None:
      ax.axhline(target, color="#d62728", ls="--", lw=1.0)
    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.grid(axis="y", alpha=0.25)
  fig.suptitle("Forward 0.4 m/s: old final vs E4 checkpoints")
  fig.savefig(PLOTS / "final_candidate_comparison.png", dpi=180)
  plt.close(fig)


def plot_hip_roll_timeseries(summaries: dict[str, dict[str, Any]]) -> None:
  keys = (
    "E0final_nominal_s42/fwd",
    "E4pre_full_nominal_s42/fwd",
    "E4final_nominal_s42/fwd",
  )
  if not all(key in summaries for key in keys):
    return
  fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
  for ax, key, title in zip(
    axes,
    keys,
    ("E0 m29999", "E4 m2500", "E4 m5000"),
    strict=True,
  ):
    d = _csv_data(ROOT / summaries[key]["file"])
    time_s = d["time_s"]
    left = d["joint_position_rad/left_hip_roll_joint"]
    right_mirrored = -d["joint_position_rad/right_hip_roll_joint"]
    ax.plot(time_s, left, lw=1.1, label="left hip-roll")
    ax.plot(time_s, right_mirrored, lw=1.1, label="-right hip-roll")
    ax.axhline(0.0, color="#999999", lw=0.7)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mirrored joint position (rad)")
    ax.grid(alpha=0.25)
    ax.legend()
  fig.suptitle("Forward 0.4 m/s hip-roll mirror agreement")
  fig.savefig(PLOTS / "hip_roll_timeseries.png", dpi=180)
  plt.close(fig)


def write_csv(summaries: dict[str, dict[str, Any]]) -> None:
  if not summaries:
    return
  fields = ["key", *next(iter(summaries.values())).keys()]
  with (OUT / "rollout_summary.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for key, row in summaries.items():
      writer.writerow({"key": key, **row})


def main() -> None:
  PLOTS.mkdir(parents=True, exist_ok=True)
  results = json.loads(RESULTS.read_text(encoding="utf-8"))
  all_series = collect_training(results)
  summaries = collect_rollouts()
  plot_training(results, all_series)
  plot_rollouts(summaries)
  plot_final_candidate(summaries)
  plot_hip_roll_timeseries(summaries)
  write_csv(summaries)
  (OUT / "analysis_summary.json").write_text(
    json.dumps(
      {
        "runs": results["runs"],
        "training": summarize_training(all_series),
        "rollouts": summaries,
      },
      ensure_ascii=False,
      indent=2,
    )
    + "\n",
    encoding="utf-8",
  )
  print(f"wrote {len(summaries)} rollout summaries to {OUT}")


if __name__ == "__main__":
  main()
