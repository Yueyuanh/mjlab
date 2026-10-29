"""Analyze the completed Pikachu V031 E0-Mirror005 training run."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util import tensor_util

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PLOTS = OUT / "plots"
RUN = ROOT / "logs/rsl_rl/pikachu_v031_e0_mirror005/2026-10-12_00-11-23_mirror005-s42"
EVENT_FILE = next(RUN.glob("events.out.tfevents.*"))
JOINT_LIMIT = 0.26

FINAL_NOMINAL = {
  clip: OUT / "metrics/final_nominal" / f"metrics_v03_{clip}_final_nominal_s42.csv"
  for clip in ("stand", "fwd", "fwd_fast", "back", "lat", "turn")
}
PROGRESSION = {
  iteration: OUT
  / f"metrics/progress_m{iteration}"
  / f"metrics_v03_fwd_m{iteration}_nominal.csv"
  for iteration in (500, 1000, 5000, 10000, 30000)
}
PROGRESSION[49999] = FINAL_NOMINAL["fwd"]
CONDITIONS = {
  "nominal": FINAL_NOMINAL["fwd"],
  "startup DR": OUT / "metrics/final_dr/metrics_v03_fwd_final_dr_s42.csv",
  "train conditions": OUT
  / "metrics/final_traincond/metrics_v03_fwd_final_traincond_s42.csv",
}
E0_FINAL = (
  ROOT / "doc/pikachu_v031_gait_improvement_20261011/metrics/"
  "E0final_nominal_s42/metrics_v03_fwd_E0final_nominal_s42.csv"
)
MANUAL_CSV = RUN / (
  "metrics_Mjlab-Velocity-Flat-Pikachu-V031-E0-Mirror005__"
  "model_49999_20261012_213919_with_target.csv"
)

TB_TAGS = (
  "Train/mean_reward",
  "Train/mean_episode_length",
  "Policy/mean_std",
  "Loss/symmetry",
  "Episode_Metrics/joint_position_rad/left_hip_roll_joint",
  "Episode_Metrics/joint_position_rad/right_hip_roll_joint",
  "Episode_Metrics/joint_target_rad/left_hip_roll_joint",
  "Episode_Metrics/joint_target_rad/right_hip_roll_joint",
  "Episode_Metrics/vel/err_yaw",
  "Episode_Reward/dof_pos_limits",
  "Episode_Reward/torque_soft_limit",
  "Episode_Reward/track_linear_velocity",
  "Episode_Reward/track_angular_velocity",
)


def load_csv(path: Path) -> dict[str, np.ndarray]:
  with path.open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
  if not rows:
    raise ValueError(f"No data rows in {path}")
  data: dict[str, np.ndarray] = {}
  for key in rows[0]:
    try:
      data[key] = np.asarray([float(row[key]) for row in rows], dtype=float)
    except (TypeError, ValueError):
      continue
  return data


def vector(data: dict[str, np.ndarray], name: str) -> np.ndarray:
  value = data.get(name)
  if value is None:
    raise KeyError(f"Missing column {name}")
  return value


def rms(value: np.ndarray) -> float:
  return float(np.sqrt(np.mean(np.square(value))))


def rollout_summary(data: dict[str, np.ndarray]) -> dict[str, float | int]:
  time_s = vector(data, "time_s")
  transient = time_s >= min(0.5, float(time_s[-1]))
  lp = vector(data, "joint_position_rad/left_hip_roll_joint")
  rp = vector(data, "joint_position_rad/right_hip_roll_joint")
  lt = vector(data, "joint_target_rad/left_hip_roll_joint")
  rt = vector(data, "joint_target_rad/right_hip_roll_joint")
  ly = vector(data, "joint_torque_nm/left_hip_roll_joint")
  ry = vector(data, "joint_torque_nm/right_hip_roll_joint")
  cmd_x = vector(data, "vel/cmd_x")
  cmd_y = vector(data, "vel/cmd_y")
  cmd_yaw = vector(data, "vel/cmd_yaw")
  act_x = vector(data, "vel/act_x")
  act_y = vector(data, "vel/act_y")
  act_yaw = vector(data, "vel/act_yaw")
  yaw = np.unwrap(vector(data, "vel/yaw_w"))

  left_abs_mean = float(np.mean(np.abs(lp[transient])))
  right_abs_mean = float(np.mean(np.abs(rp[transient])))
  left_span = float(np.ptp(lp[transient]))
  right_span = float(np.ptp(rp[transient]))
  summary: dict[str, float | int] = {
    "samples": int(len(time_s)),
    "duration_s": float(time_s[-1] - time_s[0]),
    "cmd_x_mean_m_s": float(np.mean(cmd_x[transient])),
    "cmd_y_mean_m_s": float(np.mean(cmd_y[transient])),
    "cmd_yaw_mean_rad_s": float(np.mean(cmd_yaw[transient])),
    "act_x_mean_m_s": float(np.mean(act_x[transient])),
    "act_y_mean_m_s": float(np.mean(act_y[transient])),
    "act_yaw_mean_rad_s": float(np.mean(act_yaw[transient])),
    "linear_tracking_rms_m_s": rms(
      np.linalg.norm(np.stack((cmd_x - act_x, cmd_y - act_y), axis=-1), axis=-1)[
        transient
      ]
    ),
    "yaw_tracking_rms_rad_s": rms((cmd_yaw - act_yaw)[transient]),
    "lateral_velocity_rms_m_s": rms(act_y[transient]),
    "yaw_rate_rms_rad_s": rms(act_yaw[transient]),
    "heading_change_deg": float(np.degrees(yaw[-1] - yaw[0])),
    "left_roll_mean_rad": float(np.mean(lp[transient])),
    "right_roll_mean_rad": float(np.mean(rp[transient])),
    "left_roll_abs_mean_rad": left_abs_mean,
    "right_roll_abs_mean_rad": right_abs_mean,
    "roll_abs_mean_gap_rad": abs(left_abs_mean - right_abs_mean),
    "left_roll_span_rad": left_span,
    "right_roll_span_rad": right_span,
    "roll_span_ratio_left_over_right": left_span / max(right_span, 1e-12),
    "left_target_abs_max_rad": float(np.max(np.abs(lt))),
    "right_target_abs_max_rad": float(np.max(np.abs(rt))),
    "left_target_over_limit_fraction": float(
      np.mean(np.abs(lt[transient]) > JOINT_LIMIT)
    ),
    "right_target_over_limit_fraction": float(
      np.mean(np.abs(rt[transient]) > JOINT_LIMIT)
    ),
    "left_position_near_limit_fraction": float(
      np.mean(np.abs(lp[transient]) > 0.95 * JOINT_LIMIT)
    ),
    "right_position_near_limit_fraction": float(
      np.mean(np.abs(rp[transient]) > 0.95 * JOINT_LIMIT)
    ),
    "left_roll_torque_rms_nm": rms(ly[transient]),
    "right_roll_torque_rms_nm": rms(ry[transient]),
  }
  if "policy/raw_action_abs_max" in data:
    summary["raw_action_abs_max"] = float(
      np.max(vector(data, "policy/raw_action_abs_max"))
    )
    summary["raw_action_over_one_fraction"] = float(
      np.mean(vector(data, "policy/raw_action_clip_fraction")[transient])
    )
  return summary


def moving_average(values: np.ndarray, window: int = 301) -> np.ndarray:
  if len(values) < window:
    return values.copy()
  kernel = np.ones(window, dtype=float) / window
  valid = np.convolve(values, kernel, mode="valid")
  left = window // 2
  right = len(values) - len(valid) - left
  return np.pad(valid, (left, right), mode="edge")


def load_tensorboard() -> dict[str, tuple[np.ndarray, np.ndarray]]:
  wanted = set(TB_TAGS)
  records: dict[str, list[tuple[int, float]]] = {tag: [] for tag in TB_TAGS}
  for event in EventFileLoader(str(EVENT_FILE)).Load():
    for value in event.summary.value:
      if value.tag not in wanted:
        continue
      array = tensor_util.make_ndarray(value.tensor).reshape(-1)
      if array.size:
        records[value.tag].append((int(event.step), float(array[0])))

  parsed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
  for tag, rows in records.items():
    if rows:
      parsed[tag] = (
        np.asarray([row[0] for row in rows], dtype=int),
        np.asarray([row[1] for row in rows], dtype=float),
      )

  steps = sorted({int(step) for x, _ in parsed.values() for step in x})
  by_tag = {
    tag: {int(step): float(value) for step, value in zip(x, y, strict=True)}
    for tag, (x, y) in parsed.items()
  }
  with (OUT / "training_selected_scalars.csv").open(
    "w", encoding="utf-8", newline=""
  ) as stream:
    writer = csv.writer(stream)
    writer.writerow(["iteration", *TB_TAGS])
    for step in steps:
      writer.writerow([step, *(by_tag[tag].get(step, "") for tag in TB_TAGS)])
  return parsed


def summarize_training(
  series: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
  result: dict[str, float] = {}
  for tag, (_, values) in series.items():
    tail = values[-min(1000, len(values)) :]
    result[f"{tag}:last"] = float(values[-1])
    result[f"{tag}:last_1000_mean"] = float(np.mean(tail))
  return result


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


def plot_training(series: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
  fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

  def line(ax, tag: str, label: str, color: str | None = None) -> None:
    x, y = series[tag]
    ax.plot(x, moving_average(y), label=label, color=color, lw=1.3)

  line(axes[0, 0], "Train/mean_reward", "reward", "#1f77b4")
  axes[0, 0].set(title="Mean reward (301-iteration mean)")
  line(axes[0, 1], "Train/mean_episode_length", "episode length", "#2ca02c")
  axes[0, 1].set(title="Mean episode length", ylim=(0, 1050))
  line(axes[0, 2], "Policy/mean_std", "policy std", "#9467bd")
  axes[0, 2].set(title="Policy standard deviation")

  line(
    axes[1, 0],
    "Episode_Metrics/joint_position_rad/left_hip_roll_joint",
    "left",
    "#d62728",
  )
  line(
    axes[1, 0],
    "Episode_Metrics/joint_position_rad/right_hip_roll_joint",
    "right",
    "#1f77b4",
  )
  axes[1, 0].axhline(JOINT_LIMIT, color="black", ls="--", lw=0.8)
  axes[1, 0].set(title="Episode max |hip-roll position|", ylabel="rad")
  axes[1, 0].legend()

  line(
    axes[1, 1],
    "Episode_Metrics/joint_target_rad/left_hip_roll_joint",
    "left",
    "#d62728",
  )
  line(
    axes[1, 1],
    "Episode_Metrics/joint_target_rad/right_hip_roll_joint",
    "right",
    "#1f77b4",
  )
  axes[1, 1].axhline(JOINT_LIMIT, color="black", ls="--", lw=0.8)
  axes[1, 1].set(title="Episode max |hip-roll target|", ylabel="rad")
  axes[1, 1].legend()

  line(axes[1, 2], "Loss/symmetry", "symmetry loss", "#ff7f0e")
  axes[1, 2].set(title="Mirror loss (301-iteration mean)")
  for ax in axes.flat:
    ax.set_xlabel("Iteration")
  fig.suptitle("E0-Mirror005 training curves")
  fig.savefig(PLOTS / "training_curves.png", bbox_inches="tight")
  plt.close(fig)


def plot_final_rollout(data: dict[str, np.ndarray]) -> None:
  t = vector(data, "time_s")
  lp = vector(data, "joint_position_rad/left_hip_roll_joint")
  rp = vector(data, "joint_position_rad/right_hip_roll_joint")
  lt = vector(data, "joint_target_rad/left_hip_roll_joint")
  rt = vector(data, "joint_target_rad/right_hip_roll_joint")
  fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
  axes[0, 0].plot(t, lp, label="left", color="#d62728")
  axes[0, 0].plot(t, rp, label="right", color="#1f77b4")
  axes[0, 0].axhline(JOINT_LIMIT, color="black", ls="--", lw=0.8)
  axes[0, 0].axhline(-JOINT_LIMIT, color="black", ls="--", lw=0.8)
  axes[0, 0].set(title="Hip-roll position", xlabel="Time (s)", ylabel="rad")
  axes[0, 0].legend()
  axes[0, 1].plot(t, lt, label="left target", color="#d62728")
  axes[0, 1].plot(t, rt, label="right target", color="#1f77b4")
  axes[0, 1].axhline(JOINT_LIMIT, color="black", ls="--", lw=0.8)
  axes[0, 1].axhline(-JOINT_LIMIT, color="black", ls="--", lw=0.8)
  axes[0, 1].set(title="Unclipped hip-roll targets", xlabel="Time (s)", ylabel="rad")
  axes[0, 1].legend()
  axes[1, 0].plot(t, vector(data, "vel/act_x"), label="vx")
  axes[1, 0].plot(t, vector(data, "vel/act_y"), label="vy")
  axes[1, 0].plot(t, vector(data, "vel/act_yaw"), label="yaw rate")
  axes[1, 0].axhline(0.4, color="black", ls="--", lw=0.8, label="vx command")
  axes[1, 0].set(title="Velocity response", xlabel="Time (s)")
  axes[1, 0].legend()
  px = vector(data, "vel/pos_x_w")
  py = vector(data, "vel/pos_y_w")
  points = axes[1, 1].scatter(px, py, c=t, cmap="viridis", s=8)
  axes[1, 1].set_aspect("equal", adjustable="datalim")
  axes[1, 1].set(title="World-frame path", xlabel="x (m)", ylabel="y (m)")
  fig.colorbar(points, ax=axes[1, 1], label="Time (s)")
  fig.suptitle("model_49999 nominal 0.4 m/s rollout")
  fig.savefig(PLOTS / "final_rollout_diagnostics.png", bbox_inches="tight")
  plt.close(fig)


def plot_e0_comparison(
  e0: dict[str, float | int], mirror: dict[str, float | int]
) -> None:
  labels = ("E0 m29999", "Mirror005 m49999")
  values = (
    ("Forward velocity", "act_x_mean_m_s", "m/s"),
    ("Yaw-rate RMS", "yaw_rate_rms_rad_s", "rad/s"),
    ("|Heading change|", "heading_change_deg", "deg"),
    ("Hip-roll abs-mean gap", "roll_abs_mean_gap_rad", "rad"),
    ("Mean target over-limit", "target_over", "fraction"),
    ("Raw action > 1", "raw_action_over_one_fraction", "fraction"),
  )
  records = (e0, mirror)
  fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
  for ax, (title, key, unit) in zip(axes.flat, values, strict=True):
    if key == "target_over":
      data = [
        0.5
        * (
          float(record["left_target_over_limit_fraction"])
          + float(record["right_target_over_limit_fraction"])
        )
        for record in records
      ]
    elif key == "heading_change_deg":
      data = [abs(float(record[key])) for record in records]
    else:
      data = [float(record[key]) for record in records]
    ax.bar(labels, data, color=("#7f7f7f", "#1f77b4"))
    ax.set(title=title, ylabel=unit)
    ax.tick_params(axis="x", rotation=15)
  fig.suptitle("Original E0 versus E0-Mirror005, nominal 0.4 m/s")
  fig.savefig(PLOTS / "e0_vs_mirror005.png", bbox_inches="tight")
  plt.close(fig)


def plot_progression(summaries: dict[int, dict[str, float | int]]) -> None:
  x = np.asarray(sorted(summaries))
  rows = [summaries[int(i)] for i in x]
  fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
  axes[0, 0].plot(x, [r["act_x_mean_m_s"] for r in rows], marker="o")
  axes[0, 0].axhline(0.4, color="black", ls="--", lw=0.8)
  axes[0, 0].set(title="Forward velocity", ylabel="m/s")
  axes[0, 1].plot(x, [r["roll_abs_mean_gap_rad"] for r in rows], marker="o")
  axes[0, 1].axhline(0.03, color="black", ls="--", lw=0.8)
  axes[0, 1].set(title="Hip-roll abs-mean gap", ylabel="rad")
  axes[1, 0].plot(x, [r["yaw_rate_rms_rad_s"] for r in rows], marker="o")
  axes[1, 0].axhline(0.25, color="black", ls="--", lw=0.8)
  axes[1, 0].set(title="Yaw-rate RMS", ylabel="rad/s")
  target_over = [
    0.5
    * (
      float(r["left_target_over_limit_fraction"])
      + float(r["right_target_over_limit_fraction"])
    )
    for r in rows
  ]
  axes[1, 1].plot(x, target_over, marker="o", label="target beyond 0.26 rad")
  axes[1, 1].plot(
    x,
    [r["raw_action_over_one_fraction"] for r in rows],
    marker="o",
    label="raw action > 1",
  )
  axes[1, 1].set(title="Unbounded-action utilization", ylabel="fraction")
  axes[1, 1].legend()
  for ax in axes.flat:
    ax.set_xlabel("Iteration")
  fig.suptitle("Deterministic forward-rollout progression")
  fig.savefig(PLOTS / "checkpoint_progression.png", bbox_inches="tight")
  plt.close(fig)


def plot_command_suite(summaries: dict[str, dict[str, float | int]]) -> None:
  labels = list(summaries)
  rows = [summaries[label] for label in labels]
  x = np.arange(len(labels))
  fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
  axes[0, 0].bar(x, [r["linear_tracking_rms_m_s"] for r in rows])
  axes[0, 0].set(title="Planar velocity tracking RMS", ylabel="m/s")
  axes[0, 1].bar(x, [r["yaw_tracking_rms_rad_s"] for r in rows])
  axes[0, 1].set(title="Yaw tracking RMS", ylabel="rad/s")
  axes[1, 0].bar(x, [r["roll_abs_mean_gap_rad"] for r in rows])
  axes[1, 0].axhline(0.03, color="black", ls="--", lw=0.8)
  axes[1, 0].set(title="Hip-roll abs-mean gap", ylabel="rad")
  axes[1, 1].bar(
    x,
    [
      0.5
      * (
        float(r["left_target_over_limit_fraction"])
        + float(r["right_target_over_limit_fraction"])
      )
      for r in rows
    ],
  )
  axes[1, 1].set(title="Mean hip-roll target over-limit", ylabel="fraction")
  for ax in axes.flat:
    ax.set_xticks(x, labels, rotation=20, ha="right")
  fig.suptitle("model_49999 nominal fixed-command suite")
  fig.savefig(PLOTS / "command_suite.png", bbox_inches="tight")
  plt.close(fig)


def write_rollout_table(groups: dict[str, dict[str, float | int]]) -> None:
  keys = list(next(iter(groups.values())))
  with (OUT / "rollout_summary.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["rollout", *keys])
    for name, summary in groups.items():
      writer.writerow([name, *(summary.get(key, "") for key in keys)])


def main() -> int:
  PLOTS.mkdir(parents=True, exist_ok=True)
  configure_plots()

  final = {
    name: rollout_summary(load_csv(path)) for name, path in FINAL_NOMINAL.items()
  }
  progression = {
    str(iteration): rollout_summary(load_csv(path))
    for iteration, path in PROGRESSION.items()
  }
  conditions = {
    name: rollout_summary(load_csv(path)) for name, path in CONDITIONS.items()
  }
  e0 = rollout_summary(load_csv(E0_FINAL))
  manual = rollout_summary(load_csv(MANUAL_CSV))
  series = load_tensorboard()
  training = summarize_training(series)

  plot_training(series)
  plot_final_rollout(load_csv(FINAL_NOMINAL["fwd"]))
  plot_e0_comparison(e0, final["fwd"])
  plot_progression({int(key): value for key, value in progression.items()})
  plot_command_suite(final)

  flat_rollouts: dict[str, dict[str, float | int]] = {"E0/fwd": e0}
  flat_rollouts.update({f"final/{key}": value for key, value in final.items()})
  flat_rollouts.update({f"progress/{key}": value for key, value in progression.items()})
  flat_rollouts.update({f"condition/{key}": value for key, value in conditions.items()})
  flat_rollouts["manual/final"] = manual
  write_rollout_table(flat_rollouts)

  output = {
    "run": str(RUN.relative_to(ROOT)),
    "checkpoint": "model_49999.pt",
    "joint_limit_rad": JOINT_LIMIT,
    "baseline_e0": e0,
    "final_nominal": final,
    "progression": progression,
    "conditions": conditions,
    "manual_recording": manual,
    "training": training,
  }
  (OUT / "summary.json").write_text(
    json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
  )
  print(json.dumps(output, ensure_ascii=False, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
