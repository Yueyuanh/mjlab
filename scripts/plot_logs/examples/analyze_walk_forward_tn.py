"""Analyze joint telemetry against the digitized DM motor T-N curves.

Run with::

  uv run python scripts/plot_logs/examples/analyze_walk_forward_tn.py

The generated report explicitly separates signed joint telemetry from the
first-quadrant, limited-range motor performance measurements.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METRICS = (
  REPO_ROOT
  / "pre_train"
  / "rsl_rl"
  / "pikachu_tracking"
  / "2026-08-13_17-27-46_trial-pikachu-walk-forward-dual"
  / "metrics_walk_forward_20260820_194147_with_target.csv"
)
PLOT_LOGS = REPO_ROOT / "scripts" / "plot_logs"
JOINTS = (
  "hip_pitch",
  "hip_roll",
  "hip_yaw",
  "knee",
  "ankle",
  "arm_pitch",
  "arm_roll",
)
SIDES = ("left", "right")


@dataclass(frozen=True)
class MotorSpec:
  key: str
  label: str
  curve_file: str
  joints: tuple[str, ...]
  velocity_limit_rad_s: float
  effort_limit_nm: float


MOTORS = {
  "J3507": MotorSpec(
    key="J3507",
    label="DM-J3507-2EC",
    curve_file="DM-J3507-2EC-T-N_digitized.csv",
    joints=("hip_roll", "hip_yaw", "ankle", "arm_pitch", "arm_roll"),
    velocity_limit_rad_s=10.0,
    effort_limit_nm=15.0,
  ),
  "J4310": MotorSpec(
    key="J4310",
    label="DM-J4310-2EC",
    curve_file="DM-J4310-2EC-T-N_digitized.csv",
    joints=("hip_pitch", "knee"),
    velocity_limit_rad_s=20.0,
    effort_limit_nm=10.0,
  ),
}


def read_csv_columns(path: Path) -> dict[str, np.ndarray]:
  """Read numeric CSV columns while ignoring non-numeric metadata."""
  with path.open(encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))
  if not rows:
    raise ValueError(f"CSV is empty: {path}")
  columns: dict[str, np.ndarray] = {}
  for name in rows[0]:
    try:
      columns[name] = np.asarray([float(row[name]) for row in rows], dtype=float)
    except (TypeError, ValueError):
      continue
  return columns


def motor_for_joint(joint: str) -> MotorSpec:
  return MOTORS["J4310"] if joint in MOTORS["J4310"].joints else MOTORS["J3507"]


def percentile(values: np.ndarray, quantile: float) -> float:
  return float(np.percentile(values, quantile))


def transformed_curve(
  curve: dict[str, np.ndarray],
  external_ratio: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
  """Return joint-side ``(speed_rad_s, torque_nm)`` for an ideal reduction."""
  torque = curve["torque_nm"] * external_ratio
  speed = curve["speed_rpm"] * (2.0 * np.pi / 60.0) / external_ratio
  return speed, torque


def torque_capacity(
  curve: dict[str, np.ndarray],
  speed_abs_rad_s: np.ndarray,
  external_ratio: float = 1.0,
) -> np.ndarray:
  """Estimate torque capacity by extending the measured final point to zero speed.

  This is only a diagnostic interpolation.  The source curves do not reach a
  zero-speed stall point, so the result must not be treated as a certified limit.
  """
  speed, torque = transformed_curve(curve, external_ratio)
  order = np.argsort(torque)
  torque = torque[order]
  speed = speed[order]
  speed = np.maximum.accumulate(speed[::-1])[::-1]
  torque = np.concatenate(([0.0], torque))
  speed = np.concatenate(([speed[0]], speed))
  return np.interp(
    speed_abs_rad_s,
    speed[::-1],
    torque[::-1],
    left=torque[-1],
    right=0.0,
  )


def series(
  metrics: dict[str, np.ndarray], side: str, joint: str
) -> tuple[np.ndarray, np.ndarray]:
  velocity = metrics[f"joint_velocity_rad_s/{side}_{joint}_joint"]
  torque = metrics[f"joint_torque_nm/{side}_{joint}_joint"]
  valid = np.isfinite(velocity) & np.isfinite(torque)
  return velocity[valid], torque[valid]


def summarize_joint(
  metrics: dict[str, np.ndarray],
  curves: dict[str, dict[str, np.ndarray]],
  side: str,
  joint: str,
) -> dict[str, str | float | int]:
  velocity, torque = series(metrics, side, joint)
  motor = motor_for_joint(joint)
  speed_abs = np.abs(velocity)
  torque_abs = np.abs(torque)
  direct_capacity = torque_capacity(curves[motor.key], speed_abs)
  direct_util = np.divide(
    torque_abs,
    direct_capacity,
    out=np.full_like(torque_abs, np.inf),
    where=direct_capacity > 0.0,
  )
  curve_speed, _ = transformed_curve(curves[motor.key])
  finite_direct = direct_util[np.isfinite(direct_util)]
  mapped_values: dict[str, str | float]
  if motor.key == "J3507":
    mapped_capacity = torque_capacity(curves[motor.key], speed_abs, 3.0)
    mapped_util = np.divide(
      torque_abs,
      mapped_capacity,
      out=np.full_like(torque_abs, np.inf),
      where=mapped_capacity > 0.0,
    )
    finite_mapped = mapped_util[np.isfinite(mapped_util)]
    mapped_values = {
      "j3507_3r_curve_outside_pct": float(np.mean(mapped_util > 1.0) * 100.0),
      "j3507_3r_curve_util_p95": percentile(finite_mapped, 95),
      "j3507_3r_curve_util_max": float(finite_mapped.max()),
    }
  else:
    mapped_values = {
      "j3507_3r_curve_outside_pct": "",
      "j3507_3r_curve_util_p95": "",
      "j3507_3r_curve_util_max": "",
    }
  return {
    "side": side,
    "joint": joint,
    "motor": motor.label,
    "samples": len(velocity),
    "speed_abs_p95_rad_s": percentile(speed_abs, 95),
    "speed_abs_max_rad_s": float(speed_abs.max()),
    "torque_abs_p95_nm": percentile(torque_abs, 95),
    "torque_abs_max_nm": float(torque_abs.max()),
    "negative_speed_pct": float(np.mean(velocity < 0.0) * 100.0),
    "negative_torque_pct": float(np.mean(torque < 0.0) * 100.0),
    "regenerating_pct": float(np.mean(velocity * torque < 0.0) * 100.0),
    "any_negative_coordinate_pct": float(
      np.mean((velocity < 0.0) | (torque < 0.0)) * 100.0
    ),
    "below_reference_speed_range_pct": float(
      np.mean(speed_abs < curve_speed.min()) * 100.0
    ),
    "direct_curve_outside_pct": float(np.mean(direct_util > 1.0) * 100.0),
    "direct_curve_util_p95": percentile(finite_direct, 95),
    "direct_curve_util_max": float(finite_direct.max()),
    **mapped_values,
    "config_speed_util_max": float(speed_abs.max() / motor.velocity_limit_rad_s),
    "config_torque_util_max": float(torque_abs.max() / motor.effort_limit_nm),
    "config_speed_exceed_pct": float(
      np.mean(speed_abs > motor.velocity_limit_rad_s) * 100.0
    ),
    "config_torque_exceed_pct": float(
      np.mean(torque_abs > motor.effort_limit_nm) * 100.0
    ),
  }


def aggregate_motor(
  metrics: dict[str, np.ndarray], motor: MotorSpec
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
  by_side: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
  all_velocity: list[np.ndarray] = []
  all_torque: list[np.ndarray] = []
  for joint in motor.joints:
    for side in SIDES:
      velocity, torque = series(metrics, side, joint)
      by_side[side].append(np.column_stack((velocity, torque)))
      all_velocity.append(velocity)
      all_torque.append(torque)
  return (
    np.concatenate(all_velocity),
    np.concatenate(all_torque),
    {side: np.concatenate(values) for side, values in by_side.items()},
  )


def write_summary(path: Path, rows: list[dict[str, str | float | int]]) -> None:
  with path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(
      {
        key: f"{value:.6f}" if isinstance(value, float) else value
        for key, value in row.items()
      }
      for row in rows
    )


def style_axis(axis: Axes) -> None:
  axis.grid(alpha=0.25)
  axis.spines[["top", "right"]].set_visible(False)


def plot_direct_vs_absolute(
  path: Path,
  metrics: dict[str, np.ndarray],
  curves: dict[str, dict[str, np.ndarray]],
) -> None:
  fig, axes = plt.subplots(2, 2, figsize=(14, 10))
  side_colors = {"left": "#2563eb", "right": "#dc2626"}
  for row_index, motor in enumerate(MOTORS.values()):
    _, _, by_side = aggregate_motor(metrics, motor)
    curve_speed, curve_torque = transformed_curve(curves[motor.key])
    for column_index, absolute in enumerate((False, True)):
      axis = axes[row_index, column_index]
      for side in SIDES:
        values = by_side[side]
        speed = np.abs(values[:, 0]) if absolute else values[:, 0]
        torque = np.abs(values[:, 1]) if absolute else values[:, 1]
        axis.scatter(
          speed,
          torque,
          s=5,
          alpha=0.18,
          color=side_colors[side],
          label=side.capitalize(),
          rasterized=True,
        )
      axis.plot(
        curve_speed,
        curve_torque,
        color="#111827",
        linewidth=2.4,
        label=f"{motor.label} measured curve",
      )
      axis.axhline(0.0, color="#94a3b8", linewidth=0.8)
      axis.axvline(0.0, color="#94a3b8", linewidth=0.8)
      axis.set_title(
        f"{motor.key}: {'absolute comparison' if absolute else 'current signed overlay'}"
      )
      axis.set_xlabel("|joint speed| (rad/s)" if absolute else "joint speed (rad/s)")
      axis.set_ylabel("|joint torque| (N m)" if absolute else "joint torque (N m)")
      axis.legend(fontsize=8, loc="best")
      style_axis(axis)
  fig.suptitle(
    "Why the joint operating points do not coincide with the motor T-N lines",
    fontsize=15,
  )
  fig.tight_layout()
  fig.savefig(path, dpi=180, bbox_inches="tight")
  plt.close(fig)


def plot_all_joints(
  path: Path,
  metrics: dict[str, np.ndarray],
  curves: dict[str, dict[str, np.ndarray]],
  summaries: list[dict[str, str | float | int]],
) -> None:
  fig, axes = plt.subplots(4, 2, figsize=(14, 18))
  side_colors = {"left": "#2563eb", "right": "#dc2626"}
  summary_map = {(row["side"], row["joint"]): row for row in summaries}
  for axis, joint in zip(axes.flat, JOINTS, strict=False):
    motor = motor_for_joint(joint)
    speed_curve, torque_curve = transformed_curve(curves[motor.key])
    speed_curve_rpm = speed_curve * 60.0 / (2.0 * np.pi)
    for side in SIDES:
      velocity, torque = series(metrics, side, joint)
      axis.scatter(
        np.abs(velocity) * 60.0 / (2.0 * np.pi),
        np.abs(torque),
        s=8,
        alpha=0.28,
        color=side_colors[side],
        label=side.capitalize(),
        rasterized=True,
      )
    axis.plot(
      speed_curve_rpm,
      torque_curve,
      color="#111827",
      linewidth=2.4,
      label=f"{motor.key} measured",
    )
    left = summary_map[("left", joint)]
    right = summary_map[("right", joint)]
    max_speed = max(
      float(left["speed_abs_max_rad_s"]),
      float(right["speed_abs_max_rad_s"]),
    )
    max_torque = max(
      float(left["torque_abs_max_nm"]),
      float(right["torque_abs_max_nm"]),
    )
    axis.set_title(
      f"{joint} · {motor.key}\nactual max: {max_speed:.2f} rad/s, {max_torque:.2f} N m"
    )
    axis.set_xlabel("|joint speed| (rpm)")
    axis.set_ylabel("|joint torque| (N m)")
    axis.set_xlim(left=0.0)
    axis.set_ylim(bottom=0.0)
    axis.legend(fontsize=8, loc="best")
    style_axis(axis)
  axes.flat[-1].axis("off")
  fig.suptitle(
    "Walk-forward joint operating points vs measured motor curves (absolute axes)",
    fontsize=15,
  )
  fig.tight_layout()
  fig.savefig(path, dpi=180, bbox_inches="tight")
  plt.close(fig)


def plot_j3507_3r_hypothesis(
  path: Path,
  metrics: dict[str, np.ndarray],
  curve: dict[str, np.ndarray],
) -> None:
  motor = MOTORS["J3507"]
  _, _, by_side = aggregate_motor(metrics, motor)
  fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
  side_colors = {"left": "#2563eb", "right": "#dc2626"}
  for axis, ratio in zip(axes, (1.0, 3.0), strict=True):
    for side in SIDES:
      values = by_side[side]
      axis.scatter(
        np.abs(values[:, 0]),
        np.abs(values[:, 1]),
        s=5,
        alpha=0.16,
        color=side_colors[side],
        label=side.capitalize(),
        rasterized=True,
      )
    speed, torque = transformed_curve(curve, ratio)
    axis.plot(speed, torque, color="#111827", linewidth=2.4, label="T-N curve")
    axis.set_title(
      "Dashboard direct mapping"
      if ratio == 1.0
      else "Ideal external 3R hypothesis: torque x3, speed /3"
    )
    axis.set_xlabel("|joint speed| (rad/s)")
    axis.set_ylabel("|joint torque| (N m)")
    axis.set_xlim(left=0.0)
    axis.set_ylim(bottom=0.0)
    axis.legend(fontsize=8)
    style_axis(axis)
  fig.suptitle("DM-J3507 variant/transmission ambiguity", fontsize=15)
  fig.tight_layout()
  fig.savefig(path, dpi=180, bbox_inches="tight")
  plt.close(fig)


def format_num(value: float, digits: int = 2) -> str:
  return f"{value:.{digits}f}"


def aggregate_summary(
  metrics: dict[str, np.ndarray],
  curves: dict[str, dict[str, np.ndarray]],
  motor: MotorSpec,
) -> dict[str, float | int]:
  velocity, torque, _ = aggregate_motor(metrics, motor)
  speed_abs = np.abs(velocity)
  torque_abs = np.abs(torque)
  curve_speed, curve_torque = transformed_curve(curves[motor.key])
  return {
    "samples": len(velocity),
    "speed_p95": percentile(speed_abs, 95),
    "speed_max": float(speed_abs.max()),
    "torque_p95": percentile(torque_abs, 95),
    "torque_max": float(torque_abs.max()),
    "reference_speed_min": float(curve_speed.min()),
    "reference_speed_max": float(curve_speed.max()),
    "reference_torque_min": float(curve_torque.min()),
    "reference_torque_max": float(curve_torque.max()),
    "any_negative": float(np.mean((velocity < 0.0) | (torque < 0.0)) * 100.0),
    "regenerating": float(np.mean(velocity * torque < 0.0) * 100.0),
    "speed_limit_util": float(speed_abs.max() / motor.velocity_limit_rad_s),
    "torque_limit_util": float(torque_abs.max() / motor.effort_limit_nm),
  }


def write_report(
  path: Path,
  metrics_path: Path,
  metrics: dict[str, np.ndarray],
  curves: dict[str, dict[str, np.ndarray]],
  summaries: list[dict[str, str | float | int]],
) -> None:
  aggregates = {
    key: aggregate_summary(metrics, curves, motor) for key, motor in MOTORS.items()
  }
  times = metrics["time_s"]
  j3507_3r_outside = max(
    float(row["j3507_3r_curve_outside_pct"])
    for row in summaries
    if row["motor"] == MOTORS["J3507"].label
  )
  table_rows = []
  for key in MOTORS:
    values = aggregates[key]
    table_rows.append(
      "| {key} | {samples} | {sp95} / {smax} | {tp95} / {tmax} | "
      "{rsmin}–{rsmax} | {rtmin}–{rtmax} | {negative}% | {regen}% | "
      "{slimit}% / {tlimit}% |".format(
        key=key,
        samples=values["samples"],
        sp95=format_num(float(values["speed_p95"])),
        smax=format_num(float(values["speed_max"])),
        tp95=format_num(float(values["torque_p95"])),
        tmax=format_num(float(values["torque_max"])),
        rsmin=format_num(float(values["reference_speed_min"])),
        rsmax=format_num(float(values["reference_speed_max"])),
        rtmin=format_num(float(values["reference_torque_min"])),
        rtmax=format_num(float(values["reference_torque_max"])),
        negative=format_num(float(values["any_negative"]), 1),
        regen=format_num(float(values["regenerating"]), 1),
        slimit=format_num(float(values["speed_limit_util"]) * 100.0, 1),
        tlimit=format_num(float(values["torque_limit_util"]) * 100.0, 1),
      )
    )

  report = f"""# Walk-forward 关节 T-N 与电机曲线偏差分析

## 分析对象

- Metrics：`{metrics_path.relative_to(REPO_ROOT)}`
- 采样数：{len(times)}；时段：{times[0]:.3f}–{times[-1]:.3f} s
- 电机数据：`scripts/plot_logs/TN/*_digitized.csv`
- 日志口径：`joint_velocity_rad_s` 与 `joint_torque_nm` 都是关节侧、有符号量；
  扭矩来自 MuJoCo `qfrc_actuator`。

## 结论

偏差主要不是 rpm/rad/s 换算错误，而是比较对象和可视化语义不一致：

1. 电机 CSV 只包含第一象限的正转速、正扭矩数据；当前 dashboard 直接叠加有符号
   关节数据。J3507 组有 {aggregates["J3507"]["any_negative"]:.1f}% 的点、J4310 组有
   {aggregates["J4310"]["any_negative"]:.1f}% 的点至少一个坐标为负，它们不可能贴合
   第一象限参考线。
2. 运行点处于电机曲线没有覆盖的低速区。J3507 实际绝对速度最大只有
   {aggregates["J3507"]["speed_max"]:.2f} rad/s，而参考曲线速度范围为
   {aggregates["J3507"]["reference_speed_min"]:.2f}–{aggregates["J3507"]["reference_speed_max"]:.2f}
   rad/s；J4310 对应为实际 {aggregates["J4310"]["speed_max"]:.2f} rad/s、参考
   {aggregates["J4310"]["reference_speed_min"]:.2f}–{aggregates["J4310"]["reference_speed_max"]:.2f}
   rad/s。本例所有关节样本的绝对速度都低于各自参考曲线的最低速度。
3. 两条曲线都没有延伸到零转速或零扭矩，因此是 24 V 下有限测试区间内的性能
   轨迹，不是完整的堵转/空载能力边界。动态步态运行点不应被期待落在曲线上，
   也不能仅凭“离曲线远”判断越限。
4. 软件配置限值下没有越限：J3507 的速度/扭矩最大占用分别为
   {aggregates["J3507"]["speed_limit_util"] * 100.0:.1f}% / {aggregates["J3507"]["torque_limit_util"] * 100.0:.1f}%，
   J4310 为 {aggregates["J4310"]["speed_limit_util"] * 100.0:.1f}% / {aggregates["J4310"]["torque_limit_util"] * 100.0:.1f}%。

## 汇总

| 电机组 | 点数 | 实际 abs(ω) P95 / Max (rad/s) | 实际 abs(τ) P95 / Max (N·m) | 参考速度范围 (rad/s) | 参考扭矩范围 (N·m) | 任一坐标为负 | 再生/制动点 | 软件速度/扭矩最大占用 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

“再生/制动点”按 `ω × τ < 0` 统计。第一象限曲线若用于双向执行器，至少应使用
`(|ω|, |τ|)` 比较，或镜像到四象限；再生区的真实能力仍需母线回灌和驱动器限值。

## J3507 的 3R 型号歧义

代码配置使用 `DM-J3507-2EC-3R`，总减速比注释为 21，其中外部 3R 会把上一级
输出理想映射为“扭矩 ×3、速度 ÷3”；当前 TN 文件名和图中型号是
`DM-J3507-2EC`，dashboard 则直接按 1:1 叠加。

如果 TN 曲线确实测于 3R 之前，应应用外部 3:1 变换。该假设会把参考速度范围变为
约 {aggregates["J3507"]["reference_speed_min"] / 3.0:.2f}–{aggregates["J3507"]["reference_speed_max"] / 3.0:.2f}
rad/s、扭矩范围变为 {aggregates["J3507"]["reference_torque_min"] * 3.0:.2f}–{aggregates["J3507"]["reference_torque_max"] * 3.0:.2f}
N·m；本例最差关节约有 {j3507_3r_outside:.1f}% 的样本高于这个有限测试曲线的
低速外推边界。由于原图没有明确轴侧元数据、曲线也不是完整极限包络，不能仅凭
文件名决定是否应用 3R；需要确认测试台测量轴位和具体电机变体。

J4310 也存在曲线标签 `DM-J4310-2EC` 与配置型号 `DM-J4310P-2EC-V1.1` 不完全一致
的问题，应在作为设计限值前核对规格版本。

## Dashboard 建议

1. 将运行点切换为绝对值视图，或把电机曲线镜像到四象限。
2. 把曲线标注为“24 V 实测性能曲线（有限区间）”，不要标成越限边界。
3. 默认按关节自动选择电机型号；手动选择仅用于对比。
4. 在获得包含空载点、堵转点、连续/峰值电流和再生限值的完整数据后，再计算真正
   的 T-N 越界率。
5. 明确 J3507 TN 数据位于 3R 前还是 3R 后，再决定是否做 `τ×3, ω÷3`。

## 输出文件

- `walk_forward_tn_direct_vs_absolute.png`：当前有符号叠加与绝对值比较。
- `walk_forward_tn_all_joints_absolute.png`：逐关节绝对运行点与实测曲线。
- `walk_forward_tn_j3507_3r_hypothesis.png`：J3507 直接映射和外部 3R 假设。
- `walk_forward_tn_summary.csv`：逐侧、逐关节统计明细。

### 计算限制

报告中的 `direct_curve_outside_pct` 为诊断量：它对曲线做了单调化，并将最后一个
实测点水平外推到零速。这不是厂商认证的能力边界，不应直接用于选型或安全判定。
"""
  path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
  parser.add_argument(
    "--output-dir", type=Path, default=Path(__file__).resolve().parent
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  args.metrics = args.metrics.resolve()
  args.output_dir = args.output_dir.resolve()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  metrics = read_csv_columns(args.metrics)
  curves = {
    key: read_csv_columns(PLOT_LOGS / "TN" / motor.curve_file)
    for key, motor in MOTORS.items()
  }
  summaries = [
    summarize_joint(metrics, curves, side, joint) for joint in JOINTS for side in SIDES
  ]
  outputs = {
    "summary": args.output_dir / "walk_forward_tn_summary.csv",
    "direct": args.output_dir / "walk_forward_tn_direct_vs_absolute.png",
    "joints": args.output_dir / "walk_forward_tn_all_joints_absolute.png",
    "three_r": args.output_dir / "walk_forward_tn_j3507_3r_hypothesis.png",
    "report": args.output_dir / "walk_forward_tn_analysis.md",
  }
  write_summary(outputs["summary"], summaries)
  plot_direct_vs_absolute(outputs["direct"], metrics, curves)
  plot_all_joints(outputs["joints"], metrics, curves, summaries)
  plot_j3507_3r_hypothesis(outputs["three_r"], metrics, curves["J3507"])
  write_report(outputs["report"], args.metrics, metrics, curves, summaries)
  for path in outputs.values():
    print(f"Wrote {path}")


if __name__ == "__main__":
  main()
