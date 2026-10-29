"""Summarize review metrics JSONs as one markdown table (no plotting, no images).

Usage:
  uv run python scripts/summarize_eval.py <metrics.json>... [--out table.md]

Each JSON comes from ``scripts/review_v033_policy.py``. Missing fields are
rendered as ``-`` so partially complete sweeps still produce a table.
"""

import argparse
import json
from pathlib import Path


def _fmt(value, digits=3):
  if value is None:
    return "-"
  if isinstance(value, (list, tuple)):
    return " / ".join(_fmt(v, digits) for v in value)
  if isinstance(value, float):
    return f"{value:.{digits}f}"
  return str(value)


def _knee(gait, joint):
  entry = gait.get("conditional_knee_flexion", {}).get(joint)
  if not entry:
    return "-"
  return f"{entry['swing_minus_stance_abs_mean_deg']:.1f}"


def row(path: Path) -> list[str]:
  data = json.loads(path.read_text())
  gait = data.get("gait", {})
  duty = data.get("contacts_duty") or []
  slip = data.get("contact_slip_speed_mean_m_s") or []
  spans = gait.get("leg_joint_p05_p95_span_deg") or []
  ang = gait.get("base_ang_vel_abs_mean_rad_s") or []
  foot_span = gait.get("foot_sagittal_p05_p95_span_m") or []
  # Leg joint order is right/left x (hip_pitch, knee_up, knee_mid, knee, ankle).
  knee_span = [spans[1], spans[6]] if len(spans) >= 10 else []
  return [
    path.parent.parent.name if path.parent.name == "metrics" else path.stem,
    path.name.replace("_metrics.json", ""),
    _fmt(data.get("actual_mean")),
    _fmt(data.get("mae")),
    _fmt(data.get("falls_total"), 0),
    _fmt(data.get("survival_fraction")),
    _fmt(duty),
    _fmt(slip),
    _fmt(knee_span, 1),
    _knee(gait, "right_knee_up"),
    _knee(gait, "left_knee_up"),
    _fmt(ang[0]) if ang else "-",
    _fmt(foot_span),
    _fmt(data.get("command")),
  ]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("metrics", nargs="+", type=Path)
  parser.add_argument("--out", type=Path)
  args = parser.parse_args()
  header = [
    "case",
    "clip",
    "actual vx/vy/wz",
    "MAE",
    "falls",
    "survival",
    "contact duty",
    "slip m/s",
    "knee_up span R/L deg",
    "swing-stance R deg",
    "swing-stance L deg",
    "roll rate rad/s",
    "foot sagittal span m",
    "command",
  ]
  lines = [
    "| " + " | ".join(header) + " |",
    "|" + "|".join(["---"] * len(header)) + "|",
  ]
  for path in args.metrics:
    lines.append("| " + " | ".join(row(path)) + " |")
  text = "\n".join(lines) + "\n"
  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out}")
  else:
    print(text)


if __name__ == "__main__":
  main()
