"""把同一目录下**多个 tag** 的 metrics CSV 摆成一张跨轮次对比表（Markdown）。

为什么单独一个脚本：`v03_velocity_report.py` 一次只吃一个 tag（它要出的是
「这一版策略长什么样」的报告），而 r04/r05/r06 的对比恰恰是这次交付的主线 ——
r05 修好了站姿却垮了动态、r06 用命令门控把两者拆开。这条主线要的是
**同一张表里逐行对比**，不是三份各自的报告。

数据全部来自 CSV（物理量），**不读奖励** —— 与报告 §3.4 同一条纪律：
`Episode_Reward/track_*` 是 `exp(-err²)` 的逐帧均值，站桩也能拿高分。

用法::

    cd mjlab
    uv run python scripts/v03_metrics_compare.py \
        --metrics-dir docs/pikachu_v03/metrics \
        --tags r04-best,r05-15000,r06-test \
        --out /tmp/v03_compare.md
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

CLIP_ORDER = ("stand", "fwd", "fwd_fast", "back", "lat", "turn")
CLIP_LABEL = {
  "stand": "站立",
  "fwd": "前进 0.4",
  "fwd_fast": "快走 0.8",
  "back": "后退 0.4",
  "lat": "侧移 0.3",
  "turn": "转向 0.8",
}


def _mean(xs: list[float]) -> float:
  xs = [x for x in xs if not math.isnan(x)]
  return sum(xs) / len(xs) if xs else float("nan")


def load(path: Path) -> dict[str, list[float]]:
  with path.open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
  cols: dict[str, list[float]] = {}
  for name in rows[0]:
    vals = []
    for r in rows:
      v = r.get(name)
      try:
        vals.append(float(v))  # type: ignore[arg-type]
      except (TypeError, ValueError):
        vals.append(float("nan"))
    cols[name] = vals
  return cols


def unwrap_deg(yaw: list[float]) -> float:
  """把 ±π 跳变的偏航序列解卷绕后算净转角（度）。"""
  total = 0.0
  for a, b in zip(yaw, yaw[1:], strict=False):
    d = b - a
    total += math.atan2(math.sin(d), math.cos(d))
  return math.degrees(total)


def clip_stats(cols: dict[str, list[float]], skip_frac: float = 0.125) -> dict:
  n = len(next(iter(cols.values())))
  skip = min(int(n * skip_frac), 25)
  body = slice(skip, n)

  def m(name: str) -> float:
    return _mean(cols.get(name, [float("nan")])[body])

  cmd = (m("vel/cmd_x"), m("vel/cmd_y"), m("vel/cmd_yaw"))
  act = (m("vel/act_x"), m("vel/act_y"), m("vel/act_yaw"))
  # 平面速度的「跟踪比」= 实测在命令方向上的投影 / 命令模长（与
  # eval_v03_velocity.py 同口径：只算沿命令方向的分量，不算正交漂移）。
  cmd_n = math.hypot(cmd[0], cmd[1])
  proj = (act[0] * cmd[0] + act[1] * cmd[1]) / cmd_n if cmd_n > 1e-6 else float("nan")
  ratio = proj / cmd_n if cmd_n > 1e-6 else float("nan")
  # 方向余弦：实测向量与命令向量的夹角余弦（1.0 = 方向完全正确）。
  act_n = math.hypot(act[0], act[1])
  cos = (
    (act[0] * cmd[0] + act[1] * cmd[1]) / (act_n * cmd_n)
    if act_n > 1e-6 and cmd_n > 1e-6
    else float("nan")
  )
  yaw_ratio = act[2] / cmd[2] if abs(cmd[2]) > 1e-6 else float("nan")
  yaw_net = unwrap_deg(cols.get("vel/yaw_w", []))
  px, py = cols.get("vel/pos_x_w", []), cols.get("vel/pos_y_w", [])
  net = math.hypot(px[-1] - px[0], py[-1] - py[0]) if px and py else float("nan")
  return {
    "cmd": cmd,
    "act": act,
    "ratio": ratio,
    "cos": cos,
    "yaw_ratio": yaw_ratio,
    "yaw_net_deg": yaw_net,
    "net_m": net,
    "err_xy": m("vel/err_xy_norm"),
    "err_yaw": abs(m("vel/err_yaw")),
    "sep_cm": m("foot_lateral_separation_m") * 100.0,
    "standing": m("vel/standing"),
    "slip": m("foot_slip_velocity_m_s"),
    "slide_frac": m("foot_slide_fraction"),
    "peak_power": max((abs(v) for v in cols.get("joint_power_w/left_knee_up_joint", [0.0])), default=float("nan")),
    "peak_torque_ratio": max(
      (
        abs(v)
        for k, vs in cols.items()
        if k.startswith("joint_torque_ratio/")
        for v in vs
      ),
      default=float("nan"),
    ),
  }


def _f(x: float, nd: int = 2, dash: str = "—") -> str:
  return dash if x is None or math.isnan(x) else f"{x:.{nd}f}"


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--metrics-dir", default="docs/pikachu_v03/metrics")
  ap.add_argument("--tags", required=True, help="逗号分隔，如 r04-best,r05-15000,r06-test")
  ap.add_argument("--out", default="", help="Markdown 输出路径（省略则打印到终端）")
  ap.add_argument("--title", default="跨轮次对比（同一套录制口径，逐档 8 s）")
  args = ap.parse_args()

  tags = [t.strip() for t in args.tags.split(",") if t.strip()]
  d = Path(args.metrics_dir)
  data: dict[str, dict[str, dict]] = {}
  for tag in tags:
    per_clip = {}
    for clip in CLIP_ORDER:
      p = d / f"metrics_v03_{clip}_{tag}.csv"
      if p.exists():
        per_clip[clip] = clip_stats(load(p))
    data[tag] = per_clip
    if not per_clip:
      print(f"[compare] !! {tag}: 目录里没有任何 CSV")

  L: list[str] = [f"### {args.title}", ""]
  L.append("命令与实测均在 base 系，单位 m/s（第 3 列 rad/s）。"
           "统计前丢掉每档前 12.5%（起步瞬态，上限 25 帧）。**不读任何奖励**。")
  L.append("")

  for clip in CLIP_ORDER:
    present = [t for t in tags if clip in data[t]]
    if not present:
      continue
    L.append(f"**{CLIP_LABEL.get(clip, clip)}**")
    L.append("")
    L.append("| 检查点 | 命令 (x,y,yaw) | 实测 (x,y,yaw) | 跟踪比 | 方向余弦 | 偏航比 | 净位移 (m) | 净转角 (°) | err_xy | err_yaw | 站姿开度 (cm) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in present:
      s = data[t][clip]
      L.append(
        f"| `{t}` "
        f"| {s['cmd'][0]:+.2f},{s['cmd'][1]:+.2f},{s['cmd'][2]:+.2f} "
        f"| {s['act'][0]:+.2f},{s['act'][1]:+.2f},{s['act'][2]:+.2f} "
        f"| {_f(s['ratio'])} | {_f(s['cos'])} | {_f(s['yaw_ratio'])} "
        f"| {_f(s['net_m'])} | {_f(s['yaw_net_deg'], 1)} "
        f"| {_f(s['err_xy'], 3)} | {_f(s['err_yaw'], 3)} | {_f(s['sep_cm'], 1)} |"
      )
    L.append("")

  # 全档汇总：站姿开度 + 打滑 + 峰值占用。
  L.append("**全档汇总**")
  L.append("")
  L.append("| 检查点 | 站立档开度 (cm) | 走路档开度 (cm) | 站立门控占比 | 支撑相滑移 (m/s) | 滑移帧占比 | 峰值扭矩占用 |")
  L.append("|---|---|---|---|---|---|---|")
  for t in tags:
    pc = data[t]
    if not pc:
      continue
    stand_sep = pc["stand"]["sep_cm"] if "stand" in pc else float("nan")
    walk = [pc[c]["sep_cm"] for c in CLIP_ORDER if c != "stand" and c in pc]
    walk_sep = _mean(walk)
    gate = pc["stand"]["standing"] if "stand" in pc else float("nan")
    slips = [pc[c]["slip"] for c in CLIP_ORDER if c in pc]
    fracs = [pc[c]["slide_frac"] for c in CLIP_ORDER if c in pc]
    peak_tr = max((pc[c]["peak_torque_ratio"] for c in pc), default=float("nan"))
    L.append(
      f"| `{t}` | {_f(stand_sep, 1)} | {_f(walk_sep, 1)} | {_f(gate)} "
      f"| {_f(_mean(slips), 3)} | {_f(_mean(fracs), 3)} | {_f(peak_tr)} |"
    )
  L.append("")

  text = "\n".join(L)
  if args.out:
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"[compare] -> {args.out}  ({len(text.splitlines())} 行)")
  else:
    print(text)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
