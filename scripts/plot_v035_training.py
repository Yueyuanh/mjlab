"""Plot V035 training curves from a TensorBoard run directory.

Writes ``training_curves.png`` (reward & episode length), ``training_tracking.png``
(velocity error / measured speed) and ``training_termination.png`` (fall rate).

Usage::

    uv run python scripts/plot_v035_training.py \
        logs/rsl_rl/pikachu_v035/biped_e1/<run> \
        --output logs/rsl_rl/pikachu_v035/biped_e1/<run>/curves
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
  EventAccumulator,
)

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#d9d8d3"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
CRITICAL = "#e34948"


def _style():
  plt.rcParams.update(
    {
      "figure.facecolor": SURFACE,
      "axes.facecolor": SURFACE,
      "savefig.facecolor": SURFACE,
      "font.size": 9,
      "axes.titlesize": 10,
      "axes.titleweight": "semibold",
      "axes.edgecolor": GRID,
      "axes.labelcolor": TEXT_MUTED,
      "axes.titlecolor": TEXT,
      "xtick.color": TEXT_MUTED,
      "ytick.color": TEXT_MUTED,
      "grid.color": GRID,
      "grid.linewidth": 0.6,
      "legend.frameon": False,
      "legend.fontsize": 8,
      # 中文标签需要 CJK 字形；本机 Noto Sans CJK 是 ttc 集合，matplotlib 只注册了 JP 面。
      "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
      "font.family": "sans-serif",
      "axes.unicode_minus": False,
      "lines.linewidth": 1.8,
    }
  )


def _scalars(ea, tag):
  if tag not in ea.Tags()["scalars"]:
    return None
  events = ea.Scalars(tag)
  return (
    np.array([e.step for e in events]),
    np.array([e.value for e in events]),
  )


def _smooth(values, window):
  if window <= 1 or values.size < window:
    return values
  kernel = np.ones(window) / window
  return np.convolve(values, kernel, mode="valid")


def _plot(ax, ea, tag, label, color, window, dashed=False):
  data = _scalars(ea, tag)
  if data is None:
    return False
  steps, values = data
  if window > 1 and values.size > window:
    steps = steps[window - 1 :]
    values = _smooth(values, window)
  ax.plot(steps, values, color=color, label=label, ls="--" if dashed else "-")
  return True


def _save(fig, path):
  fig.tight_layout()
  fig.savefig(path, dpi=160)
  plt.close(fig)
  print(f"wrote {path}")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("log_dir", type=Path)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--window", type=int, default=20)
  args = parser.parse_args()
  _style()
  ea = EventAccumulator(str(args.log_dir), size_guidance={"scalars": 0})
  ea.Reload()
  out = args.output or args.log_dir / "curves"
  out.mkdir(parents=True, exist_ok=True)

  fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
  _plot(axes[0], ea, "Train/mean_reward", "mean reward", SERIES[0], args.window)
  _plot(
    axes[1],
    ea,
    "Train/mean_episode_length",
    "mean episode length",
    SERIES[0],
    args.window,
  )
  axes[0].set_title("平均奖励")
  axes[1].set_title("平均回合长度 [步]")
  for ax in axes:
    ax.set_xlabel("迭代轮数")
    ax.grid(alpha=0.6)
    ax.set_axisbelow(True)
  fig.suptitle(f"训练曲线 — {args.log_dir.name}", color=TEXT)
  _save(fig, out / "training_curves.png")

  fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
  _plot(axes[0], ea, "Metrics/twist/error_vel_x", "err vx", SERIES[0], args.window)
  _plot(axes[0], ea, "Metrics/twist/error_vel_y", "err vy", SERIES[1], args.window)
  _plot(axes[0], ea, "Metrics/twist/error_vel_yaw", "err wz", SERIES[2], args.window)
  _plot(
    axes[0], ea, "Metrics/twist/error_vel_xy", "err |vxy|", SERIES[3], args.window, True
  )
  _plot(
    axes[1], ea, "Episode_Metrics/vel/cmd_x", "cmd vx", TEXT_MUTED, args.window, True
  )
  _plot(axes[1], ea, "Episode_Metrics/vel/act_x", "act vx", SERIES[0], args.window)
  _plot(
    axes[2], ea, "Episode_Metrics/vel/cmd_yaw", "cmd wz", TEXT_MUTED, args.window, True
  )
  _plot(axes[2], ea, "Episode_Metrics/vel/act_yaw", "act wz", SERIES[0], args.window)
  axes[0].set_title("速度跟踪误差 [m/s, rad/s]（含探索噪声）")
  axes[1].set_title("前进速度")
  axes[2].set_title("偏航角速度 [rad/s]")
  for ax in axes:
    ax.set_xlabel("迭代轮数")
    ax.grid(alpha=0.6)
    ax.legend()
    ax.set_axisbelow(True)
  fig.suptitle("跟踪指标（训练期遥测，带探索噪声，不可当作行为结论）", color=TEXT)
  _save(fig, out / "training_tracking.png")

  fig, ax = plt.subplots(figsize=(6.4, 3.2))
  _plot(ax, ea, "Episode_Termination/fell_over", "摔倒占比", CRITICAL, args.window)
  _plot(ax, ea, "Episode_Termination/time_out", "超时占比", SERIES[0], args.window)
  ax.set_ylim(0, 1.02)
  ax.set_xlabel("iteration")
  ax.set_ylabel("占比")
  ax.grid(alpha=0.6)
  ax.legend()
  ax.set_axisbelow(True)
  ax.set_title("终止原因构成")
  _save(fig, out / "training_termination.png")


if __name__ == "__main__":
  main()
