"""Quantify unbounded-action deployment risks for Pikachu V031 Mirror005."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PLOTS = OUT / "plots"
DATA = OUT / "data"
METRICS = ROOT / "doc/pikachu_v031_e0_mirror005_20261012/metrics/final_nominal"

JOINT_KINDS = ("hip_pitch", "hip_roll", "hip_yaw", "knee_up", "ankle")
JOINTS = tuple(
  f"{side}_{kind}_joint" for side in ("left", "right") for kind in JOINT_KINDS
)
SCALE = {
  "hip_pitch": 0.50,
  "hip_roll": 0.20,
  "hip_yaw": 0.20,
  "knee_up": 0.40,
  "ankle": 0.35,
}
POSITION_LIMIT = {
  "hip_pitch": (-2.44, 2.44),
  "hip_roll": (-0.26, 0.26),
  "hip_yaw": (-0.26, 0.26),
  "knee_up": (-0.65, 1.05),
  "ankle": (-0.79, 1.05),
}
EFFORT_LIMIT = {
  "hip_pitch": 12.5,
  "hip_roll": 12.5,
  "hip_yaw": 12.5,
  "knee_up": 12.5,
  "ankle": 3.0,
}
VELOCITY_LIMIT = {
  "hip_pitch": 12.57,
  "hip_roll": 12.57,
  "hip_yaw": 12.57,
  "knee_up": 12.57,
  "ankle": 15.71,
}
CLIPS = ("stand", "fwd", "fwd_fast", "back", "lat", "turn")


def joint_kind(name: str) -> str:
  for kind in JOINT_KINDS:
    if f"_{kind}_joint" in name:
      return kind
  raise KeyError(name)


def load_csv(path: Path) -> dict[str, np.ndarray]:
  with path.open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
  result: dict[str, np.ndarray] = {}
  for key in rows[0]:
    try:
      result[key] = np.asarray([float(row[key]) for row in rows], dtype=float)
    except (TypeError, ValueError):
      continue
  return result


def clip_path(clip: str) -> Path:
  return METRICS / f"metrics_v03_{clip}_final_nominal_s42.csv"


def transient_mask(data: dict[str, np.ndarray]) -> np.ndarray:
  time_s = data["time_s"]
  return time_s >= 0.5


def raw_actions(data: dict[str, np.ndarray]) -> np.ndarray:
  return np.stack(
    [data[f"joint_target_rad/{joint}"] / SCALE[joint_kind(joint)] for joint in JOINTS],
    axis=1,
  )


def target_violation(data: dict[str, np.ndarray]) -> np.ndarray:
  columns = []
  for joint in JOINTS:
    target = data[f"joint_target_rad/{joint}"]
    lower, upper = POSITION_LIMIT[joint_kind(joint)]
    columns.append((target < lower) | (target > upper))
  return np.stack(columns, axis=1)


def torque_ratios(data: dict[str, np.ndarray]) -> np.ndarray:
  return np.stack(
    [np.abs(data[f"joint_torque_ratio/{joint}"]) for joint in JOINTS],
    axis=1,
  )


def joint_rows(clip: str, data: dict[str, np.ndarray]) -> list[dict[str, float | str]]:
  mask = transient_mask(data)
  dt = float(np.median(np.diff(data["time_s"])))
  rows: list[dict[str, float | str]] = []
  for joint in JOINTS:
    kind = joint_kind(joint)
    target = data[f"joint_target_rad/{joint}"]
    position = data[f"joint_position_rad/{joint}"]
    velocity = np.abs(data[f"joint_velocity_rad_s/{joint}"])
    torque = np.abs(data[f"joint_torque_nm/{joint}"])
    torque_ratio = np.abs(data[f"joint_torque_ratio/{joint}"])
    raw = target / SCALE[kind]
    lower, upper = POSITION_LIMIT[kind]
    command_rate = np.abs(np.diff(target)) / dt
    command_rate_mask = mask[1:]
    target_error = target - position
    rows.append(
      {
        "clip": clip,
        "joint": joint,
        "raw_abs_max": float(np.max(np.abs(raw[mask]))),
        "raw_over_one_fraction": float(np.mean(np.abs(raw[mask]) > 1.0)),
        "target_min_rad": float(np.min(target[mask])),
        "target_max_rad": float(np.max(target[mask])),
        "target_over_hard_limit_fraction": float(
          np.mean((target[mask] < lower) | (target[mask] > upper))
        ),
        "target_error_rms_rad": float(np.sqrt(np.mean(np.square(target_error[mask])))),
        "target_error_abs_max_rad": float(np.max(np.abs(target_error[mask]))),
        "target_rate_p99_rad_s": float(
          np.percentile(command_rate[command_rate_mask], 99)
        ),
        "target_rate_max_rad_s": float(np.max(command_rate[command_rate_mask])),
        "velocity_abs_max_rad_s": float(np.max(velocity[mask])),
        "velocity_limit_rad_s": VELOCITY_LIMIT[kind],
        "torque_rms_nm": float(np.sqrt(np.mean(np.square(torque[mask])))),
        "torque_abs_max_nm": float(np.max(torque[mask])),
        "effort_limit_nm": EFFORT_LIMIT[kind],
        "torque_over_80pct_fraction": float(np.mean(torque_ratio[mask] > 0.8)),
        "torque_over_95pct_fraction": float(np.mean(torque_ratio[mask] > 0.95)),
      }
    )
  return rows


def suite_row(clip: str, data: dict[str, np.ndarray]) -> dict[str, float | str]:
  mask = transient_mask(data)
  raw = raw_actions(data)[mask]
  target_over = target_violation(data)[mask]
  torque_ratio = torque_ratios(data)[mask]
  dt = float(np.median(np.diff(data["time_s"])))
  all_target = np.stack([data[f"joint_target_rad/{joint}"] for joint in JOINTS], axis=1)
  target_rate = np.abs(np.diff(all_target, axis=0)) / dt
  max_target_rate = float(np.max(target_rate[mask[1:]]))
  return {
    "clip": clip,
    "raw_abs_max": float(np.max(np.abs(raw))),
    "raw_over_one_fraction": float(np.mean(np.abs(raw) > 1.0)),
    "steps_with_any_raw_over_one_fraction": float(
      np.mean(np.any(np.abs(raw) > 1.0, axis=1))
    ),
    "target_over_hard_limit_fraction": float(np.mean(target_over)),
    "steps_with_any_target_over_hard_limit_fraction": float(
      np.mean(np.any(target_over, axis=1))
    ),
    "torque_over_80pct_fraction": float(np.mean(torque_ratio > 0.8)),
    "steps_with_any_torque_over_80pct_fraction": float(
      np.mean(np.any(torque_ratio > 0.8, axis=1))
    ),
    "target_rate_abs_max_rad_s": max_target_rate,
  }


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
  with path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def configure_plots() -> None:
  plt.rcParams.update(
    {
      "figure.dpi": 140,
      "savefig.dpi": 180,
      "axes.grid": True,
      "grid.alpha": 0.25,
      "axes.spines.top": False,
      "axes.spines.right": False,
      "font.size": 9,
    }
  )


def short_joint(name: str) -> str:
  side = "L" if name.startswith("left") else "R"
  return f"{side}-{joint_kind(name).replace('_', ' ')}"


def plot_joint_risk(rows: list[dict[str, float | str]]) -> None:
  selected = [row for row in rows if row["clip"] == "fwd"]
  labels = [short_joint(str(row["joint"])) for row in selected]
  x = np.arange(len(selected))
  fig, axes = plt.subplots(2, 2, figsize=(13, 7.2), constrained_layout=True)

  axes[0, 0].bar(x, [float(row["raw_abs_max"]) for row in selected], color="#4c78a8")
  axes[0, 0].axhline(1.0, color="#d62728", linestyle="--", label="raw bound ±1")
  axes[0, 0].set(title="Maximum inferred raw action", ylabel="|action|")
  axes[0, 0].legend()

  axes[0, 1].bar(
    x,
    [100 * float(row["target_over_hard_limit_fraction"]) for row in selected],
    color="#f58518",
  )
  axes[0, 1].set(title="Target outside mechanical range", ylabel="% samples")

  axes[1, 0].bar(
    x,
    [float(row["target_rate_p99_rad_s"]) for row in selected],
    color="#54a24b",
    label="target slew p99",
  )
  axes[1, 0].plot(
    x,
    [float(row["velocity_limit_rad_s"]) for row in selected],
    "o--",
    color="#d62728",
    label="URDF motor velocity",
  )
  axes[1, 0].set(title="Command slew vs. motor velocity specification", ylabel="rad/s")
  axes[1, 0].legend()

  axes[1, 1].bar(
    x,
    [100 * float(row["torque_over_80pct_fraction"]) for row in selected],
    color="#e45756",
    label=">80% effort",
  )
  axes[1, 1].bar(
    x,
    [100 * float(row["torque_over_95pct_fraction"]) for row in selected],
    color="#7f3c8d",
    label=">95% effort",
  )
  axes[1, 1].set(title="Simulated effort-limit utilization", ylabel="% samples")
  axes[1, 1].legend()

  for ax in axes.flat:
    ax.set_xticks(x, labels, rotation=28, ha="right")
  fig.suptitle("Mirror005 model_49999 — 0.4 m/s nominal deployment risk")
  fig.savefig(PLOTS / "joint_risk_summary.png", bbox_inches="tight")
  plt.close(fig)


def plot_fwd_trace(data: dict[str, np.ndarray]) -> None:
  time_s = data["time_s"]
  fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, constrained_layout=True)

  for side, color in (("left", "#d62728"), ("right", "#1f77b4")):
    joint = f"{side}_hip_roll_joint"
    axes[0].plot(
      time_s,
      data[f"joint_target_rad/{joint}"],
      color=color,
      alpha=0.8,
      label=f"{side} target",
    )
    axes[0].plot(
      time_s,
      data[f"joint_position_rad/{joint}"],
      color=color,
      linestyle="--",
      linewidth=1,
      label=f"{side} actual",
    )
  axes[0].axhspan(
    -0.20, 0.20, color="#54a24b", alpha=0.08, label="recommended target envelope"
  )
  axes[0].axhline(0.26, color="black", linestyle=":", label="mechanical limit")
  axes[0].axhline(-0.26, color="black", linestyle=":")
  axes[0].set(title="Hip-roll target and actual position", ylabel="rad")
  axes[0].legend(ncol=3, fontsize=8)

  for side, color in (("left", "#d62728"), ("right", "#1f77b4")):
    joint = f"{side}_hip_yaw_joint"
    axes[1].plot(
      time_s,
      data[f"joint_target_rad/{joint}"],
      color=color,
      alpha=0.8,
      label=f"{side} target",
    )
    axes[1].plot(
      time_s,
      data[f"joint_position_rad/{joint}"],
      color=color,
      linestyle="--",
      linewidth=1,
      label=f"{side} actual",
    )
  axes[1].axhspan(-0.20, 0.20, color="#54a24b", alpha=0.08)
  axes[1].axhline(0.26, color="black", linestyle=":")
  axes[1].axhline(-0.26, color="black", linestyle=":")
  axes[1].set(title="Hip-yaw target and actual position", ylabel="rad")
  axes[1].legend(ncol=2, fontsize=8)

  for side, color in (("left", "#d62728"), ("right", "#1f77b4")):
    joint = f"{side}_ankle_joint"
    axes[2].plot(
      time_s,
      np.abs(data[f"joint_torque_ratio/{joint}"]),
      color=color,
      label=f"{side} ankle",
    )
  axes[2].axhline(0.8, color="#f58518", linestyle="--", label="80% effort")
  axes[2].axhline(1.0, color="black", linestyle=":", label="effort limit")
  axes[2].set(
    title="Ankle effort-limit utilization",
    xlabel="time (s)",
    ylabel="|torque| / limit",
  )
  axes[2].legend(ncol=4, fontsize=8)
  fig.savefig(PLOTS / "fwd_target_and_effort_trace.png", bbox_inches="tight")
  plt.close(fig)


def plot_safety_layers() -> None:
  fig, ax = plt.subplots(figsize=(14, 3.8))
  ax.set_xlim(0, 14)
  ax.set_ylim(0, 4)
  ax.axis("off")
  layers = (
    ("Policy ONNX", "finite/NaN check\nraw action telemetry", "#d9eaf7"),
    ("Bounded action", "Beta or tanh\n[-1, 1]", "#c7e9c0"),
    ("Safe target", "scale + offset\nper-joint soft limits", "#fde0a2"),
    ("Slew limiter", "velocity / acceleration\nwatchdog fallback", "#f8c4c4"),
    ("Motor control", "PD + current/torque\nthermal protection", "#d9d2e9"),
  )
  centers = np.linspace(1.4, 12.6, len(layers))
  for center, (title, body, color) in zip(centers, layers, strict=True):
    box = FancyBboxPatch(
      (center - 1.15, 1.0),
      2.3,
      2.0,
      boxstyle="round,pad=0.08",
      facecolor=color,
      edgecolor="#333333",
      linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(center, 2.45, title, ha="center", va="center", weight="bold", fontsize=10)
    ax.text(center, 1.75, body, ha="center", va="center", fontsize=8.5)
  for left, right in zip(centers[:-1], centers[1:], strict=True):
    ax.add_patch(
      FancyArrowPatch(
        (left + 1.18, 2.0),
        (right - 1.18, 2.0),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.2,
        color="#444444",
      )
    )
  ax.text(
    7,
    0.45,
    "Every layer is required; downstream torque or mechanical limits do not replace upstream command bounds.",
    ha="center",
    va="center",
    fontsize=9,
    color="#b22222",
  )
  fig.savefig(PLOTS / "deployment_safety_layers.png", bbox_inches="tight")
  plt.close(fig)


def main() -> int:
  PLOTS.mkdir(parents=True, exist_ok=True)
  DATA.mkdir(parents=True, exist_ok=True)
  configure_plots()

  datasets = {clip: load_csv(clip_path(clip)) for clip in CLIPS}
  per_joint = [row for clip, data in datasets.items() for row in joint_rows(clip, data)]
  suite = [suite_row(clip, datasets[clip]) for clip in CLIPS]
  write_csv(DATA / "joint_risk_summary.csv", per_joint)
  write_csv(DATA / "command_suite_risk.csv", suite)

  fwd_rows = [row for row in per_joint if row["clip"] == "fwd"]
  summary = {
    "source": str(clip_path("fwd").relative_to(ROOT)),
    "assumptions": {
      "default_joint_offset_rad": 0.0,
      "policy_dt_s": float(np.median(np.diff(datasets["fwd"]["time_s"]))),
      "raw_action_inference": "joint_target / configured_action_scale",
    },
    "fwd": suite[CLIPS.index("fwd")],
    "fwd_joint_risk": fwd_rows,
    "command_suite": suite,
  }
  (DATA / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
  )

  plot_joint_risk(per_joint)
  plot_fwd_trace(datasets["fwd"])
  plot_safety_layers()
  print(json.dumps(summary, ensure_ascii=False, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
