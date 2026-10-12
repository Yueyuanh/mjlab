"""v03 训练实时看板（纯文本，不读图）。

在训练**运行中**也能跑：直接读最新的 tensorboard event 文件里已经落盘的部分，
把「实时训练效果」打成一张表。全部是数值，不涉及任何图像读取。

输出六块：

  1. 头部   —— run 目录 / 迭代数 / 总步数 / 吞吐 / ETA / 已用时
  2. 核心   —— mean_reward、mean_episode_length、action_std、learning_rate
  3. 跟踪质量 —— track_linear_velocity / track_angular_velocity 及其它主要奖励项
  4. 终止分布 —— fell_over / time_out 各占多少（判断「站不站得住」）
  5. 课程与遥测 —— 命令课程阶段、actuator_spec_violation 等指标
  6. 产物   —— checkpoint 列表、最近录像、磁盘余量

用法：
    uv run python scripts/v03_train_dashboard.py                 # 最新 run
    uv run python scripts/v03_train_dashboard.py --run <目录名>   # 指定 run
    uv run python scripts/v03_train_dashboard.py --top 20         # 多打几项
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_ROOT = (
  Path(__file__).resolve().parent.parent / "logs" / "rsl_rl" / "pikachu_v03_velocity"
)

# 关心的标量（前缀 -> 该组取多少项，None 表示全取）。
CORE_KEYS = (
  "Train/mean_reward",
  "Train/mean_episode_length",
  "Train/action_std",
  "Train/learning_rate",
  "Train/mean_value_loss",
  "Train/mean_surrogate_loss",
  "Train/mean_entropy_loss",
)

GROUPS = (
  ("跟踪与主要奖励", "Episode_Reward/"),
  ("终止分布", "Episode_Termination/"),
  ("课程", "Curriculum/"),
  ("遥测指标", "Metrics/"),
)


def newest_run(explicit: str | None) -> Path:
  if explicit:
    p = Path(explicit)
    if not p.is_absolute():
      p = LOG_ROOT / explicit
    return p
  runs = sorted(
    (d for d in LOG_ROOT.iterdir() if d.is_dir()),
    key=lambda d: d.stat().st_mtime,
  )
  if not runs:
    raise SystemExit(f"没有找到任何 run：{LOG_ROOT}")
  return runs[-1]


def load(run: Path) -> EventAccumulator:
  evs = sorted(run.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
  if not evs:
    raise SystemExit(f"{run} 里没有 tensorboard event 文件")
  acc = EventAccumulator(str(evs[-1]), size_guidance={"scalars": 0})
  acc.Reload()
  return acc


def last(acc: EventAccumulator, key: str) -> tuple[int, float] | None:
  try:
    s = acc.Scalars(key)
  except KeyError:
    return None
  if not s:
    return None
  return s[-1].step, s[-1].value


def fmt(v: float) -> str:
  return f"{v:+.4f}" if abs(v) < 1e5 else f"{v:.3e}"


def bar(v: float, lo: float, hi: float, width: int = 24) -> str:
  """把 v 归一化到 lo..hi 画一条文本条（越界截断）。"""
  if hi <= lo:
    return ""
  t = max(0.0, min(1.0, (v - lo) / (hi - lo)))
  n = int(round(t * width))
  return "█" * n + "·" * (width - n)


TREND_KEYS = (
  "Train/mean_reward",
  "Train/mean_episode_length",
  "Episode_Reward/track_linear_velocity",
  "Episode_Reward/track_angular_velocity",
  "Metrics/twist/error_vel_xy",
  "Metrics/twist/error_vel_x",
  "Metrics/twist/error_vel_yaw",
  "Episode_Termination/fell_over",
)


def print_trend(acc: EventAccumulator, n: int = 8) -> None:
  """把关键标量按迭代等距抽 n 个点打成趋势表（训练是否在变好，一眼看出）。

  单点快照看不出方向：`track_linear_velocity = 1.4/2.0` 既可能是「走得不错」，
  也可能是「站着不动拿到的基线分」。要看 `error_vel_xy` 有没有往下掉。
  """
  series = {}
  for k in TREND_KEYS:
    try:
      series[k] = acc.Scalars(k)
    except KeyError:
      continue
  if not series:
    return
  ref = series.get("Train/mean_reward") or next(iter(series.values()))
  if len(ref) < 2:
    return
  idx = sorted({int(round(i * (len(ref) - 1) / (n - 1))) for i in range(n)})

  print()
  print("--- 趋势（等距抽样）---")
  head = "iter".rjust(7)
  short = {k: k.split("/", 1)[1][-18:] for k in series}
  for k in series:
    head += short[k].rjust(20)
  print(head)
  for i in idx:
    row = str(ref[i].step).rjust(7)
    for s in series.values():
      if i < len(s):
        row += f"{s[i].value:>20.4f}"
      else:
        row += " " * 20
    print(row)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", default=None, help="run 目录名（默认取最新的）")
  ap.add_argument("--top", type=int, default=14, help="每组最多打多少项")
  ap.add_argument("--trend", type=int, default=8, help="趋势表抽多少个点，0 关闭")
  args = ap.parse_args()

  run = newest_run(args.run)
  acc = load(run)

  print("=" * 78)
  print(f"Pikachu v03 训练看板   run = {run.name}")
  print("=" * 78)

  # --- 1. 头部 ---
  step = (
    acc.Scalars(CORE_KEYS[0])[-1].step
    if "Train/mean_reward" in acc.Tags()["scalars"]
    else 0
  )
  total_steps = (
    acc.Scalars("Train/total_steps")[-1].value
    if "Train/total_steps" in acc.Tags()["scalars"]
    else None
  )
  print(f"迭代        : {step}")
  if total_steps:
    print(f"总步数      : {total_steps:,.0f}")

  wall = (
    acc.Scalars("Train/time_elapsed_s")[-1].value
    if "Train/time_elapsed_s" in acc.Tags()["scalars"]
    else None
  )
  if wall and step > 0:
    per_it = wall / step
    print(f"已用时      : {wall / 3600:.2f} h   （{per_it:.2f} s/迭代）")
    print(f"ETA(3e4)    : {(30000 - step) * per_it / 3600:.2f} h")

  # --- 2. 核心 ---
  print()
  print("--- 核心 ---")
  for k in CORE_KEYS:
    r = last(acc, k)
    if r:
      print(f"  {k.split('/', 1)[1]:<22}: {fmt(r[1])}")

  # --- 趋势 ---
  if args.trend > 1:
    print_trend(acc, args.trend)

  # --- 3~5. 分组 ---
  scalars = set(acc.Tags()["scalars"])
  for title, prefix in GROUPS:
    rows = []
    for k in sorted(scalars):
      if k.startswith(prefix):
        r = last(acc, k)
        if r:
          rows.append((k[len(prefix) :], r[1]))
    if not rows:
      continue
    print()
    print(f"--- {title} ---")
    rows.sort(key=lambda kv: -abs(kv[1]))
    shown = rows[: args.top]
    width = max(len(n) for n, _ in shown)
    for name, val in shown:
      print(f"  {name:<{width}}  {fmt(val):>12}")
    if len(rows) > len(shown):
      print(f"  ...（另有 {len(rows) - len(shown)} 项，用 --top 调整）")

  # --- 6. 产物 ---
  print()
  print("--- 产物 ---")
  ckpts = sorted(
    run.glob("model_*.pt"),
    key=lambda p: int(p.stem.split("_")[1]),
  )
  if ckpts:
    print(
      f"  checkpoint : {len(ckpts)} 个，最新 {ckpts[-1].name}"
      f"（{ckpts[-1].stat().st_size / 1e6:.1f} MB）"
    )
    print(f"  全部       : {', '.join(p.stem for p in ckpts[-8:])}")
  else:
    print("  checkpoint : 尚无")

  vids = sorted(
    (run / "videos" / "train").glob("*.mp4"), key=lambda p: p.stat().st_mtime
  )
  if vids:
    total_mb = sum(p.stat().st_size for p in vids) / 1e6
    print(f"  训练录像   : {len(vids)} 段，共 {total_mb:.1f} MB，最新 {vids[-1].name}")
  else:
    print("  训练录像   : 尚无")

  du = shutil.disk_usage("/home")
  print(
    f"  磁盘 /home : 已用 {du.used / 1e9:.0f} GB / {du.total / 1e9:.0f} GB"
    f"，剩 {du.free / 1e9:.1f} GB（{du.used / du.total * 100:.0f}%）"
  )

  # 进程是否还活着
  alive = subprocess.run(
    ["pgrep", "-f", "train Mjlab-Velocity-Flat-Pikachu-V03"],
    capture_output=True,
    text=True,
  ).stdout.strip()
  print(f"  训练进程   : {'运行中' if alive else '**已退出**'}")

  # --- 训练效果速判 ---
  print()
  print("--- 速判 ---")
  r = last(acc, "Train/mean_reward")
  el = last(acc, "Train/mean_episode_length")
  if r:
    print(f"  mean_reward         {fmt(r[1])}   {bar(r[1], -5, 40)}")
  if el:
    # episode 上限 20 s × 50 Hz = 1000 控制步
    print(f"  episode_length      {el[1]:.1f}/1000   {bar(el[1], 0, 1000)}")
    if el[1] > 950:
      print("  → 基本不摔了，可以看速度跟踪")
  t = last(acc, "Episode_Reward/track_linear_velocity")
  if t:
    print(f"  track_linear_vel    {fmt(t[1])}   {bar(t[1], 0, 2.0)}")
  to = last(acc, "Episode_Termination/time_out")
  fo = last(acc, "Episode_Termination/fell_over")
  if to and fo:
    tot = to[1] + fo[1]
    if tot > 0:
      print(f"  跑满整段占比        {to[1] / tot * 100:.1f}%")


if __name__ == "__main__":
  main()
