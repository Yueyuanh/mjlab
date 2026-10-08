"""把 v03 各轮训练的曲线导成 markdown 表，供离线 review（不读图）。

tensorboard 的曲线适合「看」，不适合「引用」。报告里要写「r02 的
`error_vel_xy` 从 0.59 降到 0.46」这种话时，需要一个可以复制、可以 diff、
可以在 git 里留痕的文本形式 —— 这个脚本就是干这个的。

**采样策略**：不导全部点（几万个点写进 markdown 没人看），按**对数间隔**
取 ~14 个采样点：早期密（变化快）、后期疏（趋于平稳）。

用法::

    cd mjlab
    uv run python scripts/v03_curve_md.py                 # 导全部已知 run
    uv run python scripts/v03_curve_md.py --out docs/pikachu_v03/reports/curve.md
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE = Path("logs/rsl_rl/pikachu_v03_velocity")

# 轮次标签 -> run 目录名（按时间顺序）。
RUNS: dict[str, str] = {
  "r01": "2026-10-01_12-17-25_v03-flat-r01",
  "r02": "2026-10-01_12-36-03_v03-flat-r02",
  "r03": "2026-10-01_14-11-08_v03-flat-r03",
  "r04": "2026-10-01_14-48-28_v03-flat-r04",
}

# 展示名 -> tensorboard tag。
KEYS: dict[str, str] = {
  "mean_reward": "Train/mean_reward",
  "ep_len": "Train/mean_episode_length",
  "err_xy": "Metrics/twist/error_vel_xy",
  "err_yaw": "Metrics/twist/error_vel_yaw",
  "track_lin": "Episode_Reward/track_linear_velocity",
  "track_ang": "Episode_Reward/track_angular_velocity",
  "air_time": "Episode_Reward/air_time",
  "alt_air_time": "Episode_Reward/alternating_air_time",
  "fell_over": "Episode_Termination/fell_over",
}

# 这些指标「越小越好」，表里用 ↓ 标出来，免得 review 时看反。
LOWER_IS_BETTER = {"err_xy", "err_yaw", "fell_over"}


def load_run(run: str) -> dict[str, list[tuple[int, float]]]:
  """读一个 run 的全部标量。目录不存在或没有 event 文件时返回空。"""
  d = BASE / run
  if not d.is_dir():
    return {}
  files = glob.glob(os.path.join(d, "**", "events.out.tfevents.*"), recursive=True)
  if not files:
    return {}
  # 同一个 run 可能有多个 event 文件（重启过），全部读进来按 step 合并。
  merged: dict[str, list[tuple[int, float]]] = {}
  for f in sorted(files):
    ea = EventAccumulator(f, size_guidance={"scalars": 0})
    ea.Reload()
    for tag in ea.Tags()["scalars"]:
      merged.setdefault(tag, []).extend((e.step, e.value) for e in ea.Scalars(tag))
  for tag in merged:
    merged[tag].sort(key=lambda t: t[0])
  return merged


def sample(series: list[tuple[int, float]], n: int) -> list[tuple[int, float]]:
  """按**对数间隔**取 n 个点：早期密、后期疏。"""
  if len(series) <= n:
    return series
  # 在 log(1+step) 上等距取，再映射回索引。
  import math

  lo, hi = math.log1p(series[0][0]), math.log1p(series[-1][0])
  want = [math.expm1(lo + (hi - lo) * i / (n - 1)) for i in range(n)]
  out, j = [], 0
  for w in want:
    while j + 1 < len(series) and series[j][0] < w:
      j += 1
    out.append(series[j])
  # 去重（相邻 want 可能落到同一个点上）
  dedup: list[tuple[int, float]] = []
  for p in out:
    if not dedup or dedup[-1][0] != p[0]:
      dedup.append(p)
  return dedup


def fmt(v: float, lower_better: bool) -> str:
  s = f"{v:.3f}" if abs(v) < 1000 else f"{v:.0f}"
  return s


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default=None, help="写到文件；不给我就打到 stdout")
  ap.add_argument("--points", type=int, default=14, help="每个 run 取多少个采样点")
  args = ap.parse_args()

  data = {label: load_run(run) for label, run in RUNS.items()}
  data = {k: v for k, v in data.items() if v}
  if not data:
    print("没有任何 run 有 event 文件")
    return 1

  lines: list[str] = []
  lines.append("# Pikachu v03 训练曲线（文本快照）")
  lines.append("")
  lines.append(
    "> 由 `scripts/v03_curve_md.py` 生成，按对数间隔采样 "
    f"{args.points} 个点。"
  )
  lines.append("> 标记 ↓ 的指标**越小越好**，其余越大越好。")
  lines.append("")

  for label, series_all in data.items():
    n_iter = 0
    for tag in series_all.values():
      if tag:
        n_iter = max(n_iter, tag[-1][0])
    lines.append(f"## {label}（跑到 iter {n_iter}）")
    lines.append("")
    lines.append("| iter | " + " | ".join(
      f"{k}{' ↓' if k in LOWER_IS_BETTER else ''}" for k in KEYS
    ) + " |")
    lines.append("|---|" + "---|" * len(KEYS))

    # 以 err_xy 的 step 为骨架对齐各指标（它们都是同一个 log_interval 打的）。
    ref_tag = KEYS["err_xy"]
    if ref_tag not in series_all:
      ref_tag = next(iter(series_all))
    ref = sample(series_all[ref_tag], args.points)

    for step, _ in ref:
      cells = []
      for name, tag in KEYS.items():
        s = series_all.get(tag)
        if not s:
          cells.append("—")
          continue
        # 取 <= step 的最后一个点
        val = None
        for st, v in s:
          if st <= step:
            val = v
          else:
            break
        cells.append(fmt(val, name in LOWER_IS_BETTER) if val is not None else "—")
      lines.append(f"| {step} | " + " | ".join(cells) + " |")
    lines.append("")

  text = "\n".join(lines)
  if args.out:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"已写入 {out}（{len(text)} 字节）")
  else:
    print(text)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
