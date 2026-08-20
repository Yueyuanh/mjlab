"""Check whether a reference motion npz is physically feasible to track.

参考动作（``csv_to_npz_pikachu`` 产物，纯运动学）在进入 tracking 训练前，
先用本脚本做一次离线可行性体检：脚是否真的落地、支撑相是否滑脚、关节角/角速度
是否在硬件限位与电机规格内。参考本身不可达（悬空/拖脚），策略再调奖励也学不出
钉地，所以这是训练前的第一道关卡。

输出 JSON 报告 + 终端打印，并给出 ``feasible`` 结论与 ``issues`` 列表。

Example:
  uv run python -m mjlab.scripts.check_reference_feasibility \
      --motion-file src/mjlab/mocap/npz/pikachu_walk_npz/pikachu_walk_0.185_0.0_0.0.npz

  uv run python -m mjlab.scripts.check_reference_feasibility \
      --motion-file src/mjlab/mocap/npz/Pikachu_Walk2_S1.npz --output /tmp/report.json
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import tyro

from mjlab.asset_zoo.robots.pikachu.pikachu_constants import (
  ACTUATOR_J3507,
  ACTUATOR_J4310,
)

# Pikachu 14-dof 关节顺序（与 PIKACHU_JOINT_NAMES 一致，见 csv_to_npz_pikachu.py）。
PIKACHU_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_joint",
  "left_arm_pitch_joint",
  "left_arm_roll_joint",
  "right_arm_pitch_joint",
  "right_arm_roll_joint",
)

# 关节位置硬限位（rad），来源 xmls/mjcf_14dof_lite/defaults.xml。
JOINT_POS_LIMITS: dict[str, tuple[float, float]] = {
  "left_hip_pitch_joint": (-2.9, 0.698),
  "right_hip_pitch_joint": (-0.698, 2.9),
  "left_hip_roll_joint": (-0.09, 0.26),
  "right_hip_roll_joint": (-0.09, 0.26),
  "left_hip_yaw_joint": (-0.09, 0.26),
  "right_hip_yaw_joint": (-0.09, 0.26),
  "left_knee_joint": (-1.57, 0),
  "right_knee_joint": (0, 1.57),
  "left_ankle_joint": (-1.05, 0.52),
  "right_ankle_joint": (-0.52, 1.05),
  "left_arm_pitch_joint": (-3.14, 1.05),
  "right_arm_pitch_joint": (-1.05, 3.14),
  "left_arm_roll_joint": (0.0, 1.57),
  "right_arm_roll_joint": (-1.57, 0.0),
}

# 执行器分组（关节名正则 -> 规格）。来源 pikachu_constants.py：
#   J4310（髋 pitch/膝）velocity_limit=30 rad/s, effort_limit=10 N·m
#   J3507（髋 roll/yaw/踝/臂）velocity_limit=10 rad/s, effort_limit=15 N·m
ACTUATOR_GROUPS: tuple[tuple[str, float, float], ...] = (
  (
    r".*(hip_pitch|knee)_joint",
    ACTUATOR_J4310.velocity_limit,
    ACTUATOR_J4310.effort_limit,
  ),
  (
    r".*(hip_roll|hip_yaw|ankle|arm_pitch|arm_roll)_joint",
    ACTUATOR_J3507.velocity_limit,
    ACTUATOR_J3507.effort_limit,
  ),
)

# npz body 索引（Pikachu 17 body 顺序，见 analyze：5=left_ankle, 10=right_ankle）。
LEFT_ANKLE_IDX = 5
RIGHT_ANKLE_IDX = 10


@dataclass
class CheckReferenceFeasibilityConfig:
  """Configuration for reference motion feasibility check."""

  motion_file: str
  """Path to the reference motion npz (csv_to_npz_pikachu output)."""
  output: str | None = None
  """Optional JSON path to write the report to."""
  sole_offset: float = 0.0487
  """Ankle-to-sole vertical offset (m), calibrated from the clean rmg walk npz."""
  hover_height: float = 0.012
  """Foot-sole height (m) above which a foot is considered hovering."""
  slip_threshold: float = 0.2
  """Stance horizontal foot speed (m/s) above which we call it sliding."""
  slip_frac_limit: float = 0.05
  """Max allowed fraction of stance time sliding; above -> not feasible."""
  root_z_drift_limit: float = 0.15
  """Max allowed root x/y drift (m/s) trend tolerance for the verdict."""


@dataclass
class FootReport:
  name: str
  sole_z_min: float
  sole_z_median: float
  hover_frac: float
  stance_frac: float
  stance_slip_mean: float
  stance_slip_p95: float
  stance_slip_p99: float
  stance_slip_max: float
  slip_frac_gt_threshold: float


@dataclass
class JointReport:
  name: str
  pos_min: float
  pos_max: float
  pos_range: tuple[float, float]
  pos_exceed_frac: float
  vel_max: float
  vel_limit: float
  vel_exceed_frac: float


@dataclass
class Report:
  motion_file: str
  fps: float
  num_frames: int
  duration_s: float
  root_z_min: float
  root_z_max: float
  root_drift_m: float
  feet: list[FootReport] = field(default_factory=list)
  joints: list[JointReport] = field(default_factory=list)
  feasible: bool = True
  issues: list[str] = field(default_factory=list)


def _actuator_spec(joint_name: str) -> tuple[float, float]:
  for pattern, v_limit, e_limit in ACTUATOR_GROUPS:
    if re.fullmatch(pattern, joint_name):
      return v_limit, e_limit
  msg = f"No actuator spec for joint {joint_name}"
  raise ValueError(msg)


def _foot_report(
  ankle_z: np.ndarray,
  vh: np.ndarray,
  name: str,
  cfg: CheckReferenceFeasibilityConfig,
) -> FootReport:
  sole_z = ankle_z - cfg.sole_offset
  # Planted plateau: the lowest ankle-z band (数据自适应的钉地平台期)。用百分位
  # 而非绝对阈值，toe-off/heel-strike 时踝部抬离平台，不会被误判成滑脚。
  plateau_th = float(np.percentile(ankle_z, 30)) + 1e-3
  planted = ankle_z < plateau_th
  planted_frac = float(planted.mean())
  slip = vh[planted] if planted_frac > 0 else np.zeros((0,), dtype=np.float32)
  return FootReport(
    name=name,
    sole_z_min=float(sole_z.min()),
    sole_z_median=float(np.median(sole_z)),
    hover_frac=float((sole_z > cfg.hover_height).mean()),
    stance_frac=planted_frac,
    stance_slip_mean=float(slip.mean()) if slip.size else 0.0,
    stance_slip_p95=float(np.percentile(slip, 95)) if slip.size else 0.0,
    stance_slip_p99=float(np.percentile(slip, 99)) if slip.size else 0.0,
    stance_slip_max=float(slip.max()) if slip.size else 0.0,
    slip_frac_gt_threshold=float((slip > cfg.slip_threshold).mean())
    if slip.size
    else 0.0,
  )


def check_reference(cfg: CheckReferenceFeasibilityConfig) -> Report:
  """Run the feasibility checks and assemble a report."""
  data = np.load(cfg.motion_file)
  fps = float(np.ravel(data["fps"])[0])
  dt = 1.0 / fps
  body_pos = data["body_pos_w"]
  joint_pos = data["joint_pos"]
  joint_vel = data["joint_vel"]
  num_frames = body_pos.shape[0]

  root = body_pos[:, 0, :]
  root_drift = float(np.hypot(root[-1, 0] - root[0, 0], root[-1, 1] - root[0, 1]))

  vel = np.gradient(body_pos, axis=0) / dt
  report = Report(
    motion_file=cfg.motion_file,
    fps=fps,
    num_frames=num_frames,
    duration_s=float(num_frames * dt),
    root_z_min=float(root[:, 2].min()),
    root_z_max=float(root[:, 2].max()),
    root_drift_m=root_drift,
  )

  # Feet: groundedness + stance sliding.
  for name, idx in (("left", LEFT_ANKLE_IDX), ("right", RIGHT_ANKLE_IDX)):
    ankle_z = body_pos[:, idx, 2]
    vh = np.hypot(vel[:, idx, 0], vel[:, idx, 1])
    report.feet.append(_foot_report(ankle_z, vh, name, cfg))

  # Joints: position / velocity vs hardware limits.
  for i, joint_name in enumerate(PIKACHU_JOINT_NAMES):
    pos = joint_pos[:, i]
    vel_j = joint_vel[:, i]
    lo, hi = JOINT_POS_LIMITS[joint_name]
    v_limit, _ = _actuator_spec(joint_name)
    report.joints.append(
      JointReport(
        name=joint_name,
        pos_min=float(pos.min()),
        pos_max=float(pos.max()),
        pos_range=(lo, hi),
        pos_exceed_frac=float(((pos < lo) | (pos > hi)).mean()),
        vel_max=float(np.abs(vel_j).max()),
        vel_limit=v_limit,
        vel_exceed_frac=float((np.abs(vel_j) > v_limit).mean()),
      )
    )

  # Verdict.
  for foot in report.feet:
    if foot.sole_z_median > cfg.hover_height:
      report.feasible = False
      report.issues.append(
        f"{foot.name} foot median sole height {foot.sole_z_median * 100:.1f}cm "
        "> hover threshold — reference walks with the foot off the ground"
      )
    if foot.slip_frac_gt_threshold > cfg.slip_frac_limit:
      report.feasible = False
      report.issues.append(
        f"{foot.name} foot slides {foot.slip_frac_gt_threshold * 100:.0f}% of stance "
        f"(> {cfg.slip_threshold} m/s) — reference itself drags the foot"
      )
  for joint in report.joints:
    if joint.pos_exceed_frac > 1e-6:
      report.issues.append(
        f"{joint.name} exceeds joint limits {joint.pos_exceed_frac * 100:.2f}% of frames"
      )
    if joint.vel_exceed_frac > 1e-6:
      report.feasible = False
      report.issues.append(
        f"{joint.name} exceeds velocity spec {joint.vel_exceed_frac * 100:.2f}% of frames "
        f"(peak {joint.vel_max:.1f} rad/s vs limit {joint.vel_limit:.1f})"
      )
  return report


def _print_report(report: Report) -> None:
  print(f"motion: {report.motion_file}")
  print(
    f"fps={report.fps:.0f} frames={report.num_frames} duration={report.duration_s:.1f}s"
  )
  print(
    f"root z [{report.root_z_min:.3f}, {report.root_z_max:.3f}]  drift xy={report.root_drift_m:.3f}m"
  )
  print("\nfeet (sole height / stance sliding):")
  for foot in report.feet:
    print(
      f"  {foot.name:<5} sole z min/med {foot.sole_z_min * 100:6.1f}/{foot.sole_z_median * 100:5.1f}cm "
      f"hover {foot.hover_frac * 100:4.0f}% | stance {foot.stance_frac * 100:3.0f}% "
      f"slip mean/p95/p99/max {foot.stance_slip_mean:.3f}/{foot.stance_slip_p95:.3f}/"
      f"{foot.stance_slip_p99:.3f}/{foot.stance_slip_max:.3f} m/s "
      f"gt-thr {foot.slip_frac_gt_threshold * 100:.1f}%"
    )
  print("\njoints (position / velocity vs limits):")
  for joint in report.joints:
    if joint.pos_exceed_frac > 1e-6 or joint.vel_exceed_frac > 1e-6:
      print(
        f"  {joint.name:<24} pos [{joint.pos_min:+.2f},{joint.pos_max:+.2f}] "
        f"in {joint.pos_range} exceed {joint.pos_exceed_frac * 100:.1f}% | "
        f"vel peak {joint.vel_max:.1f}/{joint.vel_limit:.0f} rad/s exceed "
        f"{joint.vel_exceed_frac * 100:.1f}%"
      )
  print(f"\nfeasible: {report.feasible}")
  for issue in report.issues:
    print(f"  [issue] {issue}")


def main() -> None:
  """Entry point."""
  cfg = tyro.cli(CheckReferenceFeasibilityConfig)
  report = check_reference(cfg)
  _print_report(report)
  if cfg.output is not None:
    out = Path(cfg.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(report), indent=2))
    print(f"\n[INFO] report written to {out}")


if __name__ == "__main__":
  main()
