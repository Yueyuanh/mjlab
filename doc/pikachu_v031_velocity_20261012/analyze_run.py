"""为 ``2026-10-12_20-05-34`` 这次 V031-T6 训练生成图、表、视频与 summary。

与 ``doc/pikachu_v031_velocity_20261010/analyze_run.py`` **同一套口径**（列契约、
``JOINT_LIMIT``、移动平均窗口、tfevents 读法都逐字沿用），差别只在数据源：

- 旧包分析的是**外部提供的**一段异常 rollout CSV + 5 个受控回放；
- 本包分析的是**本次训练自己录的**受控回放（2 checkpoint × 3 条件 × stand/fwd）、
  本次 ``diag_v031_rollout`` 的基座姿态遥测，以及 7 个 checkpoint 的
  ``eval_v031_symmetry`` 结果（本次 3 个 + 上次 4 个，用于跨 run 对照）。

产物：``plots/*.png`` / ``videos/*.mp4`` / ``summary.json`` /
``training_selected_scalars.csv``。

从仓库根目录运行::

    uv run python doc/pikachu_v031_velocity_20261012/analyze_run.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_file_loader import (  # noqa: E402
  EventFileLoader,
)
from tensorboard.util import tensor_util  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PLOTS = OUT / "plots"
VIDEOS = OUT / "videos"
RUN = ROOT / "logs/rsl_rl/pikachu_v031_velocity/2026-10-12_20-05-34"
EVENT_FILE = RUN / ("events.out.tfevents.1791806737.finnox-Legion-R7000-APH9.1183624.0")
# 与 eval_v031_symmetry.py:59 的 HIP_ROLL_LIMIT 同值；报告里的"越界率"就是它。
JOINT_LIMIT = 0.26

# 受控回放：checkpoint × 条件，档位固定 fwd(+0.40 m/s)。键是图上的行标签。
REPLAY_DIRS = {
  "m1000 nominal": "model_1000_nominal",
  "m1000 DR": "model_1000_dr",
  "m1000 train cond.": "model_1000_train",
  "m4999 nominal": "model_4999_nominal",
  "m4999 DR": "model_4999_dr",
  "m4999 train cond.": "model_4999_train",
}
# 遥测视频/诊断图用哪一段回放（本次训练最终 checkpoint、保留启动 DR）。
PRIMARY_REPLAY = "model_4999_dr"

EVAL_FILES = (
  ("newT6 m1000", "newT6_model_1000.json"),
  ("newT6 m3000", "newT6_model_3000.json"),
  ("newT6 m4999", "newT6_model_4999.json"),
  ("oldT6 m1000", "oldT6_model_1000.json"),
  ("oldT6 m3000", "oldT6_model_3000.json"),
  ("oldT6 m5000", "oldT6_model_5000.json"),
  ("oldT6 m29999", "oldT6_model_29999.json"),
)

# 验收线（与 20261010/REPORT.md 的「验收线对账」一节逐条对应）。
ACCEPTANCE = {
  "fwd_heading_drift_deg": 10.0,
  "target_violation": 0.0,
  "hr_d_abs": 0.03,
  "yaw_rate_rms_rad_s": 0.25,
}

TB_TAGS = (
  "Train/mean_reward",
  "Train/mean_episode_length",
  "Policy/mean_std",
  "Episode_Metrics/joint_position_rad/left_hip_roll_joint",
  "Episode_Metrics/joint_position_rad/right_hip_roll_joint",
  "Episode_Metrics/joint_target_rad/left_hip_roll_joint",
  "Episode_Metrics/joint_target_rad/right_hip_roll_joint",
  "Episode_Metrics/vel/err_yaw",
  "Episode_Reward/dof_pos_limits",
  "Episode_Reward/torque_soft_limit",
  "Episode_Reward/track_linear_velocity",
  "Episode_Reward/track_angular_velocity",
  "Episode_Reward/alternating_air_time",
  "Episode_Reward/walking_stance_width",
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
  """与旧包**逐字相同**的摘要口径，便于两份报告的数字直接对齐。"""
  time_s = vector(data, "time_s")
  transient = time_s >= min(0.5, float(time_s[-1]))
  lp = vector(data, "joint_position_rad/left_hip_roll_joint")
  rp = vector(data, "joint_position_rad/right_hip_roll_joint")
  lt = vector(data, "joint_target_rad/left_hip_roll_joint")
  rt = vector(data, "joint_target_rad/right_hip_roll_joint")
  ly = vector(data, "joint_torque_nm/left_hip_roll_joint")
  ry = vector(data, "joint_torque_nm/right_hip_roll_joint")
  yaw = np.unwrap(vector(data, "vel/yaw_w"))
  act_y = vector(data, "vel/act_y")
  act_yaw = vector(data, "vel/act_yaw")
  act_x = vector(data, "vel/act_x")
  cmd_x = vector(data, "vel/cmd_x")
  px = vector(data, "vel/pos_x_w")
  py = vector(data, "vel/pos_y_w")

  def span(value: np.ndarray) -> float:
    return float(np.max(value) - np.min(value))

  return {
    "samples": int(len(time_s)),
    "duration_s": float(time_s[-1] - time_s[0]),
    "cmd_x_mean_m_s": float(np.mean(cmd_x[transient])),
    "act_x_mean_m_s": float(np.mean(act_x[transient])),
    "act_y_mean_m_s": float(np.mean(act_y[transient])),
    "act_y_rms_m_s": rms(act_y[transient]),
    "act_yaw_mean_rad_s": float(np.mean(act_yaw[transient])),
    "act_yaw_rms_rad_s": rms(act_yaw[transient]),
    "heading_change_deg": float(np.degrees(yaw[-1] - yaw[0])),
    "net_displacement_m": float(np.hypot(px[-1] - px[0], py[-1] - py[0])),
    "left_roll_min_rad": float(np.min(lp)),
    "left_roll_max_rad": float(np.max(lp)),
    "left_roll_mean_rad": float(np.mean(lp[transient])),
    "left_roll_span_rad": span(lp[transient]),
    "right_roll_min_rad": float(np.min(rp)),
    "right_roll_max_rad": float(np.max(rp)),
    "right_roll_mean_rad": float(np.mean(rp[transient])),
    "right_roll_span_rad": span(rp[transient]),
    "roll_span_ratio_left_over_right": span(lp[transient])
    / max(span(rp[transient]), 1e-12),
    # 与 eval 脚本同定义的"同相/反相"分解：hr_d 是共模（应≈0），hr_s 是差模。
    "hr_d_mean_rad": float(np.mean(lp[transient] + rp[transient]) / 2.0),
    "hr_s_mean_rad": float(np.mean(lp[transient] - rp[transient]) / 2.0),
    "hr_d_absmax_rad": float(np.max(np.abs((lp + rp) / 2.0))),
    "left_target_abs_max_rad": float(np.max(np.abs(lt))),
    "right_target_abs_max_rad": float(np.max(np.abs(rt))),
    "left_target_over_limit_fraction": float(np.mean(np.abs(lt) > JOINT_LIMIT)),
    "right_target_over_limit_fraction": float(np.mean(np.abs(rt) > JOINT_LIMIT)),
    # 与 eval 脚本同口径：hip_roll **和** hip_yaw 四条目标里任一越界即算一次。
    # 只数 hip_roll 会低估（eval 那边一直是 0.85–0.99），两份报告的数字要对得上。
    "target_violation": float(
      np.mean(
        np.any(
          [
            np.abs(vector(data, f"joint_target_rad/{side}_{axis}_joint")) > JOINT_LIMIT
            for side in ("left", "right")
            for axis in ("hip_roll", "hip_yaw")
          ],
          axis=0,
        )
      )
    ),
    "left_position_near_limit_fraction": float(
      np.mean(np.abs(lp) > 0.95 * JOINT_LIMIT)
    ),
    "right_position_near_limit_fraction": float(
      np.mean(np.abs(rp) > 0.95 * JOINT_LIMIT)
    ),
    "left_target_tracking_rmse_rad": rms(lt - lp),
    "right_target_tracking_rmse_rad": rms(rt - rp),
    "left_roll_torque_rms_nm": rms(ly[transient]),
    "right_roll_torque_rms_nm": rms(ry[transient]),
  }


def eval_summary(path: Path) -> dict[str, float | int | bool]:
  """把一份 ``eval_v031_symmetry.py`` 的 JSON 压成一行跨 checkpoint 可比的数字。"""
  payload = json.loads(path.read_text(encoding="utf-8"))
  rows = payload["rows"]
  by_cmd: dict[str, list[dict]] = {}
  for row in rows:
    by_cmd.setdefault(row["command"], []).append(row)

  def mean_of(cmd: str, key: str) -> float:
    values = [float(r[key]) for r in by_cmd.get(cmd, []) if r[key] == r[key]]
    return float(np.mean(values)) if values else float("nan")

  def max_of(cmd: str, key: str) -> float:
    values = [float(r[key]) for r in by_cmd.get(cmd, []) if r[key] == r[key]]
    return float(np.max(values)) if values else float("nan")

  # 平移命令（要换步的那四条）上的最坏值——诊断信号就藏在这里。
  translating = ("fwd", "back", "left", "right")
  return {
    "checkpoint": payload.get("checkpoint", ""),
    "clip_actions": payload.get("clip_actions"),
    "seconds": payload.get("seconds"),
    "any_fell": bool(any(r["fell"] for r in rows)),
    "fwd_heading_drift_deg": mean_of("fwd", "heading_drift_deg_per_m")
    * mean_of("fwd", "path_len_m"),
    "fwd_yaw_drift_deg": mean_of("fwd", "yaw_drift_deg"),
    "fwd_target_violation": mean_of("fwd", "target_violation"),
    "max_target_violation": float(np.max([float(r["target_violation"]) for r in rows])),
    "stand_target_violation": mean_of("stand", "target_violation"),
    "fwd_hr_p2p_ratio": mean_of("fwd", "hr_p2p_ratio"),
    "hr_d_absmax_rad": max_of("fwd", "hr_d"),
    "hr_d_absmax_all_cmds_rad": float(np.max([abs(float(r["hr_d"])) for r in rows])),
    "fwd_hr_d_rad": mean_of("fwd", "hr_d"),
    "stand_hr_d_rad": mean_of("stand", "hr_d"),
    "fwd_action_absmax": max_of("fwd", "action_absmax"),
    "fwd_action_clip_frac": mean_of("fwd", "action_clip_frac"),
    "turn_yaw_rate_rms_rad_s": float(
      np.mean([mean_of(c, "yaw_rate_rms") for c in ("turnL", "turnR")])
    ),
    "stand_yaw_rate_rms_rad_s": mean_of("stand", "yaw_rate_rms"),
    "translating_heading_drift_deg": float(
      np.mean(
        [
          mean_of(c, "heading_drift_deg_per_m") * mean_of(c, "path_len_m")
          for c in translating
        ]
      )
    ),
    # mirror_equiv 的原始 dict 里有 per_dim（10 维），报告只用一个数：ratio。
    "mirror_equiv_ratio": (payload.get("mirror_equiv") or {}).get("ratio"),
    "agg_straight": payload.get("agg_straight") or {},
  }


def moving_average(values: np.ndarray, window: int = 301) -> np.ndarray:
  if len(values) < window:
    return values.copy()
  kernel = np.ones(window, dtype=float) / window
  valid = np.convolve(values, kernel, mode="valid")
  pad_left = window // 2
  pad_right = len(values) - len(valid) - pad_left
  return np.pad(valid, (pad_left, pad_right), mode="edge")


def configure_plot_style() -> None:
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


def plot_training_curves(
  series: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
  fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), constrained_layout=True)

  def plot_tag(ax, tag: str, label: str, color: str | None = None) -> None:
    x, y = series[tag]
    ax.plot(x, moving_average(y), label=label, color=color, lw=1.4)

  plot_tag(axes[0, 0], "Train/mean_reward", "mean reward", "#1f77b4")
  axes[0, 0].set(title="Training return (301-iteration mean)", ylabel="return")

  left_pos = "Episode_Metrics/joint_position_rad/left_hip_roll_joint"
  right_pos = "Episode_Metrics/joint_position_rad/right_hip_roll_joint"
  plot_tag(axes[0, 1], left_pos, "left max |q|", "#d62728")
  plot_tag(axes[0, 1], right_pos, "right max |q|", "#1f77b4")
  axes[0, 1].axhline(JOINT_LIMIT, color="black", ls="--", lw=0.8)
  axes[0, 1].set(title="Episode hip-roll position metric", ylabel="rad")
  axes[0, 1].legend()

  left_target = "Episode_Metrics/joint_target_rad/left_hip_roll_joint"
  right_target = "Episode_Metrics/joint_target_rad/right_hip_roll_joint"
  plot_tag(axes[1, 0], left_target, "left max |target|", "#d62728")
  plot_tag(axes[1, 0], right_target, "right max |target|", "#1f77b4")
  axes[1, 0].axhline(JOINT_LIMIT, color="black", ls="--", lw=0.8)
  axes[1, 0].set(title="Episode hip-roll target metric", ylabel="rad")
  axes[1, 0].legend()

  plot_tag(axes[1, 1], "Episode_Metrics/vel/err_yaw", "yaw error", "#ff7f0e")
  plot_tag(axes[1, 1], "Episode_Reward/dof_pos_limits", "joint-limit reward", "#9467bd")
  plot_tag(
    axes[1, 1], "Episode_Reward/torque_soft_limit", "torque-limit reward", "#2ca02c"
  )
  axes[1, 1].set(title="Yaw error and limit penalties")
  axes[1, 1].legend()

  for ax in axes.flat:
    ax.set_xlabel("Iteration")
  fig.suptitle("New T6 run (2026-10-12_20-05-34): saved TensorBoard event file")
  fig.savefig(PLOTS / "training_curves.png", bbox_inches="tight")
  plt.close(fig)

  summary: dict[str, float] = {}
  for tag, (_, values) in series.items():
    tail = values[-min(1000, len(values)) :]
    summary[f"{tag}:last"] = float(values[-1])
    summary[f"{tag}:last_1000_mean"] = float(np.mean(tail))

  lx, ly = series[left_pos]
  rx, ry = series[right_pos]
  if np.array_equal(lx, rx):
    gap = np.abs(ly - ry)
    summary["hip_roll_position_lr_gap:last_1000_mean_rad"] = float(
      np.mean(gap[-min(1000, len(gap)) :])
    )
    summary["hip_roll_position_lr_gap:max_rad"] = float(np.max(gap))
  return summary


def plot_controlled_comparison(
  summaries: dict[str, dict[str, float | int]],
) -> None:
  labels = list(summaries)
  x = np.arange(len(labels))
  width = 0.36
  fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), constrained_layout=True)

  left_span = [float(summaries[k]["left_roll_span_rad"]) for k in labels]
  right_span = [float(summaries[k]["right_roll_span_rad"]) for k in labels]
  axes[0, 0].bar(x - width / 2, left_span, width, label="left")
  axes[0, 0].bar(x + width / 2, right_span, width, label="right")
  axes[0, 0].set(title="Hip-roll peak-to-peak span (fwd 0.40 m/s)", ylabel="rad")
  axes[0, 0].legend()

  left_over = [
    100 * float(summaries[k]["left_target_over_limit_fraction"]) for k in labels
  ]
  right_over = [
    100 * float(summaries[k]["right_target_over_limit_fraction"]) for k in labels
  ]
  axes[0, 1].bar(x - width / 2, left_over, width, label="left")
  axes[0, 1].bar(x + width / 2, right_over, width, label="right")
  axes[0, 1].set(title="Target outside +/-0.26 rad", ylabel="samples (%)")
  axes[0, 1].legend()

  yaw_rms = [float(summaries[k]["act_yaw_rms_rad_s"]) for k in labels]
  lat_rms = [float(summaries[k]["act_y_rms_m_s"]) for k in labels]
  axes[1, 0].bar(x - width / 2, yaw_rms, width, label="yaw RMS (rad/s)")
  axes[1, 0].bar(x + width / 2, lat_rms, width, label="lateral RMS (m/s)")
  axes[1, 0].axhline(ACCEPTANCE["yaw_rate_rms_rad_s"], color="black", ls="--", lw=0.8)
  axes[1, 0].set(title="Uncommanded motion RMS")
  axes[1, 0].legend()

  act_x = [float(summaries[k]["act_x_mean_m_s"]) for k in labels]
  cmd_x = [float(summaries[k]["cmd_x_mean_m_s"]) for k in labels]
  axes[1, 1].bar(x - width / 2, cmd_x, width, label="command")
  axes[1, 1].bar(x + width / 2, act_x, width, label="actual")
  axes[1, 1].set(title="Forward tracking", ylabel="m/s")
  axes[1, 1].legend()

  for ax in axes.flat:
    ax.set_xticks(x, labels, rotation=20, ha="right")
  fig.suptitle("New T6 controlled replays: checkpoint x randomization condition")
  fig.savefig(PLOTS / "controlled_replay_comparison.png", bbox_inches="tight")
  plt.close(fig)


def plot_symmetry_acceptance(
  evals: dict[str, dict[str, float | int | bool]],
) -> None:
  """7 个 checkpoint 对 4 条验收线的对账图；阈值画成线/带，不靠肉眼比数。"""
  labels = list(evals)
  x = np.arange(len(labels))
  new_count = sum(1 for name in labels if name.startswith("newT6"))
  colors = ["#1f77b4" if i < new_count else "#7f7f7f" for i in range(len(labels))]

  fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), constrained_layout=True)

  ax = axes[0, 0]
  values = [float(evals[k]["fwd_heading_drift_deg"]) for k in labels]
  ax.bar(x, values, 0.6, color=colors)
  ax.axhline(
    ACCEPTANCE["fwd_heading_drift_deg"],
    color="black",
    ls="--",
    lw=1.0,
    label=f"acceptance < {ACCEPTANCE['fwd_heading_drift_deg']:.0f} deg",
  )
  ax.set(title="fwd 8 s heading drift", ylabel="deg")
  ax.legend()

  ax = axes[0, 1]
  values = [100 * float(evals[k]["fwd_target_violation"]) for k in labels]
  ax.bar(x, values, 0.6, color=colors)
  ax.axhline(
    ACCEPTANCE["target_violation"],
    color="black",
    ls="--",
    lw=1.0,
    label=f"acceptance = {ACCEPTANCE['target_violation']:.0f}%",
  )
  ax.set(title="fwd hip-roll target outside +/-0.26 rad", ylabel="samples (%)")
  ax.legend()

  ax = axes[1, 0]
  values = [float(evals[k]["hr_d_absmax_all_cmds_rad"]) for k in labels]
  ax.bar(x, values, 0.6, color=colors)
  ax.axhline(
    ACCEPTANCE["hr_d_abs"] * 2.0,
    color="black",
    ls="--",
    lw=1.0,
    label=f"acceptance |qL+qR|/2 < {ACCEPTANCE['hr_d_abs'] * 2:.2f} rad",
  )
  ax.set(title="worst in-phase hip-roll hr_d", ylabel="|qL+qR|/2 (rad)")
  ax.legend()

  ax = axes[1, 1]
  values = [float(evals[k]["fwd_hr_p2p_ratio"]) for k in labels]
  ax.bar(x, values, 0.6, color=colors)
  ax.axhspan(0.8, 1.25, color="#2ca02c", alpha=0.10, label="acceptance 0.8-1.25")
  ax.axhline(1.0, color="black", lw=0.7)
  ax.set(title="fwd hip-roll p2p ratio std(L)/std(R)", ylabel="ratio")
  ax.legend()

  for ax in axes.flat:
    ax.set_xticks(x, labels, rotation=20, ha="right")
  fig.suptitle(
    "Acceptance-line reconciliation (blue = this run, grey = 2026-10-10 run)"
  )
  fig.savefig(PLOTS / "symmetry_acceptance.png", bbox_inches="tight")
  plt.close(fig)


def plot_replay_diagnostics(data: dict[str, np.ndarray]) -> None:
  time_s = vector(data, "time_s")
  lp = vector(data, "joint_position_rad/left_hip_roll_joint")
  rp = vector(data, "joint_position_rad/right_hip_roll_joint")
  lt = vector(data, "joint_target_rad/left_hip_roll_joint")
  rt = vector(data, "joint_target_rad/right_hip_roll_joint")
  act_y = vector(data, "vel/act_y")
  act_yaw = vector(data, "vel/act_yaw")
  px = vector(data, "vel/pos_x_w")
  py = vector(data, "vel/pos_y_w")

  fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
  ax = axes[0, 0]
  ax.plot(time_s, lp, label="left position", color="#d62728")
  ax.plot(time_s, rp, label="right position", color="#1f77b4")
  ax.axhline(JOINT_LIMIT, color="black", ls="--", lw=0.8)
  ax.axhline(-JOINT_LIMIT, color="black", ls="--", lw=0.8)
  ax.set(title="Recorded hip-roll position", xlabel="Time (s)", ylabel="rad")
  ax.legend(loc="upper right")

  ax = axes[0, 1]
  ax.plot(time_s, lt, label="left target", color="#d62728")
  ax.plot(time_s, rt, label="right target", color="#1f77b4")
  ax.axhspan(-JOINT_LIMIT, JOINT_LIMIT, color="#2ca02c", alpha=0.08)
  ax.axhline(JOINT_LIMIT, color="black", ls="--", lw=0.8)
  ax.axhline(-JOINT_LIMIT, color="black", ls="--", lw=0.8)
  ax.set(title="Unclipped policy targets", xlabel="Time (s)", ylabel="rad")
  ax.legend(loc="upper right")

  ax = axes[1, 0]
  ax.plot(time_s, act_y, label="lateral velocity (m/s)", color="#9467bd")
  ax.plot(time_s, act_yaw, label="yaw rate (rad/s)", color="#ff7f0e")
  ax.axhline(0.0, color="black", lw=0.7)
  ax.set(title="Zero-command lateral/yaw response", xlabel="Time (s)")
  ax.legend(loc="upper right")

  ax = axes[1, 1]
  points = ax.scatter(px, py, c=time_s, cmap="viridis", s=9)
  ax.scatter(px[0], py[0], color="green", marker="o", label="start")
  ax.scatter(px[-1], py[-1], color="red", marker="x", label="end")
  ax.set_aspect("equal", adjustable="datalim")
  ax.set(title="World-frame path", xlabel="x (m)", ylabel="y (m)")
  ax.legend(loc="best")
  fig.colorbar(points, ax=ax, label="Time (s)")
  fig.suptitle(f"Controlled replay {PRIMARY_REPLAY}: fwd 0.40 m/s, 8 s, seed 42")
  fig.savefig(PLOTS / "replay_diagnostics.png", bbox_inches="tight")
  plt.close(fig)


def make_telemetry_video(data: dict[str, np.ndarray]) -> None:
  time_s = vector(data, "time_s")
  lp = vector(data, "joint_position_rad/left_hip_roll_joint")
  rp = vector(data, "joint_position_rad/right_hip_roll_joint")
  lt = vector(data, "joint_target_rad/left_hip_roll_joint")
  rt = vector(data, "joint_target_rad/right_hip_roll_joint")
  act_y = vector(data, "vel/act_y")
  act_yaw = vector(data, "vel/act_yaw")
  px = vector(data, "vel/pos_x_w")
  py = vector(data, "vel/pos_y_w")

  fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.4), constrained_layout=True)
  fig.suptitle(f"New T6 telemetry: {PRIMARY_REPLAY} fwd 0.40 m/s")
  ax_pos, ax_target, ax_vel, ax_path = axes.flat
  pos_lines = (
    ax_pos.plot([], [], color="#d62728", label="left")[0],
    ax_pos.plot([], [], color="#1f77b4", label="right")[0],
  )
  target_lines = (
    ax_target.plot([], [], color="#d62728", label="left")[0],
    ax_target.plot([], [], color="#1f77b4", label="right")[0],
  )
  vel_lines = (
    ax_vel.plot([], [], color="#9467bd", label="v_y")[0],
    ax_vel.plot([], [], color="#ff7f0e", label="yaw rate")[0],
  )
  path_line = ax_path.plot([], [], color="#333333", lw=1.4)[0]
  path_point = ax_path.plot([], [], "o", color="#d62728", ms=5)[0]

  for ax in (ax_pos, ax_target):
    ax.axhline(JOINT_LIMIT, color="black", ls="--", lw=0.7)
    ax.axhline(-JOINT_LIMIT, color="black", ls="--", lw=0.7)
    ax.set_xlim(time_s[0], time_s[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("rad")
    ax.legend(loc="upper right")
  ax_pos.set_ylim(min(lp.min(), rp.min(), -0.3), max(lp.max(), rp.max(), 0.3))
  ax_pos.set_title("Hip-roll position")
  ax_target.set_ylim(min(lt.min(), rt.min(), -0.3), max(lt.max(), rt.max(), 0.3))
  ax_target.set_title("Unclipped target")
  ax_vel.set_xlim(time_s[0], time_s[-1])
  ax_vel.set_ylim(min(act_y.min(), act_yaw.min()), max(act_y.max(), act_yaw.max()))
  ax_vel.set(title="Uncommanded motion", xlabel="Time (s)")
  ax_vel.legend(loc="upper right")
  pad_x = max(0.05, 0.05 * np.ptp(px))
  pad_y = max(0.05, 0.05 * np.ptp(py))
  ax_path.set_xlim(px.min() - pad_x, px.max() + pad_x)
  ax_path.set_ylim(py.min() - pad_y, py.max() + pad_y)
  ax_path.set_aspect("equal", adjustable="box")
  ax_path.set(title="World path", xlabel="x (m)", ylabel="y (m)")
  clock = fig.text(0.5, 0.01, "", ha="center")

  dest = VIDEOS / "controlled_replay_telemetry.mp4"
  writer = imageio.get_writer(
    dest, fps=50, codec="libx264", quality=8, macro_block_size=None
  )
  try:
    for index in range(len(time_s)):
      end = index + 1
      pos_lines[0].set_data(time_s[:end], lp[:end])
      pos_lines[1].set_data(time_s[:end], rp[:end])
      target_lines[0].set_data(time_s[:end], lt[:end])
      target_lines[1].set_data(time_s[:end], rt[:end])
      vel_lines[0].set_data(time_s[:end], act_y[:end])
      vel_lines[1].set_data(time_s[:end], act_yaw[:end])
      path_line.set_data(px[:end], py[:end])
      path_point.set_data([px[index]], [py[index]])
      clock.set_text(f"t = {time_s[index]:.2f} s")
      fig.canvas.draw()
      frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
      writer.append_data(frame)
  finally:
    writer.close()
    plt.close(fig)


def stream_tensorboard() -> dict[str, tuple[np.ndarray, np.ndarray]]:
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
    if not rows:
      continue
    parsed[tag] = (
      np.asarray([row[0] for row in rows], dtype=int),
      np.asarray([row[1] for row in rows], dtype=float),
    )

  steps = sorted({int(step) for pair in parsed.values() for step in pair[0]})
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


def main() -> int:
  PLOTS.mkdir(parents=True, exist_ok=True)
  VIDEOS.mkdir(parents=True, exist_ok=True)
  configure_plot_style()

  primary = OUT / f"metrics/{PRIMARY_REPLAY}/metrics_v03_fwd_{PRIMARY_REPLAY}.csv"
  source = load_csv(primary)
  plot_replay_diagnostics(source)
  make_telemetry_video(source)

  summaries: dict[str, dict[str, float | int]] = {}
  for label, folder in REPLAY_DIRS.items():
    path = OUT / f"metrics/{folder}/metrics_v03_fwd_{folder}.csv"
    if path.exists():
      summaries[label] = rollout_summary(load_csv(path))
  if summaries:
    plot_controlled_comparison(summaries)

  evals = {
    label: eval_summary(OUT / "symmetry" / name)
    for label, name in EVAL_FILES
    if (OUT / "symmetry" / name).exists()
  }
  if evals:
    plot_symmetry_acceptance(evals)

  tb_series = stream_tensorboard()
  training_summary = plot_training_curves(tb_series)

  output = {
    "run": str(RUN.relative_to(ROOT)),
    "event_file": EVENT_FILE.name,
    "joint_limit_rad": JOINT_LIMIT,
    "primary_replay": PRIMARY_REPLAY,
    "acceptance_lines": ACCEPTANCE,
    "replays": summaries,
    "symmetry": evals,
    "training": training_summary,
  }
  (OUT / "summary.json").write_text(
    json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
  )
  print(f"wrote {OUT / 'summary.json'}")
  print(f"  replays: {len(summaries)}  evals: {len(evals)}  tags: {len(tb_series)}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
