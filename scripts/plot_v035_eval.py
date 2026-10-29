"""Plot V035 evaluation series (matplotlib, static PNG for the report).

Reads ``eval.json`` + ``eval.npz`` written by ``scripts/eval_v035.py`` and
writes four figures next to them:

* ``fig1_speed_tracking.png``  command vs measured mean per clip,
* ``fig2_velocity_trace.png``  per-clip velocity traces (mean ± std),
* ``fig3_height_support.png``  base height and support count,
* ``fig4_joint_envelope.png``  joint travel vs soft limits,
* ``fig5_actuator_usage.png``  torque/speed scatter against motor specs.

Usage::

    uv run python scripts/plot_v035_eval.py reports/v035/biped reports/v035/crawl
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#d9d8d3"
COMMAND = "#8f8e88"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
CRITICAL = "#e34948"

CLIP_ORDER = ("stand", "forward", "backward", "lateral", "turn")
CLIP_ZH = {
  "stand": "站立",
  "forward": "前进",
  "backward": "后退",
  "lateral": "侧移",
  "turn": "转向",
}
AXES = ("vx", "vy", "wz")


def _style():
  plt.rcParams.update(
    {
      "figure.facecolor": SURFACE,
      "axes.facecolor": SURFACE,
      "savefig.facecolor": SURFACE,
      "font.size": 9,
      "axes.titlesize": 10,
      "axes.titleweight": "semibold",
      "axes.labelsize": 9,
      "axes.edgecolor": GRID,
      "axes.labelcolor": TEXT_MUTED,
      "axes.titlecolor": TEXT,
      "xtick.color": TEXT_MUTED,
      "ytick.color": TEXT_MUTED,
      "grid.color": GRID,
      "grid.linewidth": 0.6,
      "legend.frameon": False,
      "legend.fontsize": 8,
      "lines.linewidth": 2,
      # 中文标签需要 CJK 字形；本机 Noto Sans CJK 是 ttc 集合，matplotlib 只注册了 JP 面。
      "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
      "font.family": "sans-serif",
      "axes.unicode_minus": False,
    }
  )


def _save(fig, path: Path):
  fig.tight_layout()
  fig.savefig(path, dpi=160)
  plt.close(fig)
  print(f"wrote {path}")


def load(eval_dir: Path):
  result = json.loads((eval_dir / "eval.json").read_text())
  series = np.load(eval_dir / "eval.npz")
  clips = [c for c in CLIP_ORDER if c in result["clips"]]
  return result, series, clips


def fig_speed(result, clips, out: Path):
  fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
  x = np.arange(len(clips))
  width = 0.36
  for axis, (ax, label) in enumerate(zip(axes, AXES, strict=True)):
    command = [result["clips"][c]["command"][axis] for c in clips]
    measured = [result["clips"][c]["mean_velocity"][axis] for c in clips]
    ax.bar(x - width / 2, command, width, color=COMMAND, label="命令")
    ax.bar(x + width / 2, measured, width, color=SERIES[0], label="实测")
    for xi, value in zip(x, measured, strict=True):
      # 零附近的柱子如果按柱顶/柱底放标签会压到 x 轴刻度文字上，统一抬到零线以上。
      tiny = abs(value) < 0.02
      ax.text(
        xi + width / 2,
        0.002 if tiny else value + (0.008 if value >= 0 else -0.008),
        f"{value:+.3f}",
        ha="center",
        va="bottom" if (tiny or value >= 0) else "top",
        fontsize=7,
        color=TEXT,
        # 相邻柱子的数值标签在 wz 面板会挤在一起，加个同底色背景板隔开。
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.4},
      )
    ax.axhline(0, color=TEXT_MUTED, lw=0.8)
    ax.set_xticks(x, [CLIP_ZH.get(c, c) for c in clips], rotation=20)
    ax.set_title(label)
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    if axis == 0:
      ax.legend(loc="upper right")
  fig.suptitle(
    f"命令跟踪：每档 {result['seconds']:.0f} s 的平均值", color=TEXT, fontsize=11
  )
  _save(fig, out)


def fig_trace(result, series, clips, out: Path):
  fig, axes = plt.subplots(3, len(clips), figsize=(2.3 * len(clips), 5.6), sharex=True)
  for col, clip in enumerate(clips):
    t = series[f"{clip}/t"]
    velocity = series[f"{clip}/velocity"]
    if velocity.ndim == 2:  # Older eval files stored only env 0.
      velocity = velocity[:, None, :]
    command = result["clips"][clip]["command"]
    for row, axis in enumerate(range(3)):
      ax = axes[row, col]
      ax.axhline(command[axis], color=COMMAND, ls="--", lw=1.2)
      ax.fill_between(
        t,
        velocity[:, axis].mean(1) - velocity[:, axis].std(1),
        velocity[:, axis].mean(1) + velocity[:, axis].std(1),
        color=SERIES[0],
        alpha=0.18,
        lw=0,
      )
      ax.plot(t, velocity[:, axis].mean(1), color=SERIES[0], lw=1.6)
      ax.axhline(0, color=GRID, lw=0.8)
      ax.grid(alpha=0.6)
      ax.set_axisbelow(True)
      if row == 0:
        ax.set_title(CLIP_ZH.get(clip, clip))
      if col == 0:
        ax.set_ylabel(("vx [m/s]", "vy [m/s]", "wz [rad/s]")[row])
      if row == 2:
        ax.set_xlabel("时间 [s]")
  fig.suptitle("速度时间序列：实线 = 跨环境均值，色带 = ±1 标准差", color=TEXT)
  _save(fig, out)


def fig_height(result, series, clips, out: Path, stand_height):
  fig, axes = plt.subplots(2, len(clips), figsize=(2.3 * len(clips), 3.8), sharex=True)
  for col, clip in enumerate(clips):
    t = series[f"{clip}/t"]
    height = series[f"{clip}/base_height"]
    support = series[f"{clip}/support_found"].sum(axis=1)
    ax = axes[0, col]
    ax.axhline(stand_height, color=COMMAND, ls="--", lw=1.2)
    ax.plot(t, height, color=SERIES[0], lw=1.4)
    ax.grid(alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title(CLIP_ZH.get(clip, clip))
    if col == 0:
      ax.set_ylabel("机身高度 [m]")
    ax = axes[1, col]
    ax.plot(t, support, color=SERIES[1], lw=1.4, drawstyle="steps-post")
    ax.set_ylim(-0.2, support.max() + 0.6 if support.size else 1)
    ax.grid(alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("时间 [s]")
    if col == 0:
      ax.set_ylabel("支撑接触数")
  fig.suptitle("机身高度（虚线 = 站立关键帧高度）与支撑接触数", color=TEXT)
  _save(fig, out)


def fig_envelope(result, series, clips, out: Path, clip="forward"):
  joints = result["joint_names"]
  limits = np.asarray(result["soft_joint_pos_limits"])
  if clip not in clips:
    clip = clips[0]
  position = series[f"{clip}/joint_pos"]
  fig, ax = plt.subplots(figsize=(8.6, 6.4))
  y = np.arange(len(joints))
  low, high = position.min(0), position.max(0)
  for yi in y:
    ax.plot(
      [limits[yi, 0], limits[yi, 1]], [yi, yi], color=GRID, lw=7, solid_capstyle="butt"
    )
    ax.plot(
      [low[yi], high[yi]], [yi, yi], color=SERIES[0], lw=4, solid_capstyle="round"
    )
  ax.set_yticks(y, joints)
  ax.invert_yaxis()
  ax.grid(axis="x", alpha=0.6)
  ax.set_axisbelow(True)
  ax.set_xlabel("关节位置 [rad] — 灰 = 软限位，蓝 = 实测最小…最大值")
  ax.set_title(f"关节行程（{CLIP_ZH.get(clip, clip)}档）")
  _save(fig, out)


def fig_actuator(result, series, clips, out: Path, clip="forward"):
  if clip not in clips:
    clip = clips[0]
  joints = result["joint_names"]
  specs = result["joint_specs"]
  torque = series[f"{clip}/joint_torque"]
  velocity = series[f"{clip}/joint_vel"]
  groups = [
    ("膝关节 4340", [j for j in joints if "knee" in j], SERIES[0]),
    (
      "髋/踝/臂 4310",
      [j for j in joints if "knee" not in j and not re.search(r"arm_yaw|elbow", j)],
      SERIES[1],
    ),
    (
      "臂偏航/肘 3507",
      [j for j in joints if re.search(r"arm_yaw|elbow", j)],
      SERIES[2],
    ),
  ]
  fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
  for ax, (title, names, color) in zip(axes, groups, strict=True):
    effort = max(specs[n]["effort_limit"] for n in names)
    speed = max(specs[n]["velocity_limit"] for n in names)
    for name in names:
      i = joints.index(name)
      ax.scatter(velocity[:, i], torque[:, i], s=4, alpha=0.25, color=color, lw=0)
    ax.axvline(speed, color=CRITICAL, lw=1, ls="--")
    ax.axvline(-speed, color=CRITICAL, lw=1, ls="--")
    ax.axhline(effort, color=CRITICAL, lw=1, ls="--")
    ax.axhline(-effort, color=CRITICAL, lw=1, ls="--")
    ax.set_xlim(-speed * 1.15, speed * 1.15)
    ax.set_ylim(-effort * 1.15, effort * 1.15)
    ax.grid(alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title(f"{title}\n±{effort:.0f} N·m, ±{speed:.1f} rad/s")
    ax.set_xlabel("关节速度 [rad/s]")
  axes[0].set_ylabel("关节力矩 [N·m]")
  fig.suptitle(
    f"电机工作点（{CLIP_ZH.get(clip, clip)}档，红色虚线 = 电机规格）", color=TEXT
  )
  _save(fig, out)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("eval_dirs", nargs="+", type=Path)
  parser.add_argument("--stand-height", type=float, default=0.22957707433403704)
  parser.add_argument("--clip", default="forward")
  args = parser.parse_args()
  _style()
  for directory in args.eval_dirs:
    result, series, clips = load(directory)
    if not clips:
      continue
    fig_speed(result, clips, directory / "fig1_speed_tracking.png")
    fig_trace(result, series, clips, directory / "fig2_velocity_trace.png")
    fig_height(
      result, series, clips, directory / "fig3_height_support.png", args.stand_height
    )
    fig_envelope(
      result, series, clips, directory / "fig4_joint_envelope.png", args.clip
    )
    fig_actuator(
      result, series, clips, directory / "fig5_actuator_usage.png", args.clip
    )


if __name__ == "__main__":
  main()
