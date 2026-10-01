"""v03 训练**单行**监控（给 Monitor 用）。

``v03_train_dashboard.py`` 是多行表格，适合人看；但 Monitor 把 stdout 的
**每一行**都变成一条通知，多行输出会刷屏。这里只吐一行，把「该不该管」的
判据压进去：

    iter=3000 ep_len=1000 rew=+118.4 | err_xy 0.68→0.41 ↓ | err_yaw 0.31→0.22 ↓
    | air_time=+0.412 alt=+1.83 | track_lin=1.62/2.0 | fell=0.00 | 0.72s/it

关键在 **err_xy 的箭头**：它带箭头才能一眼看出修复有没有生效 ——
r01 那次是「平在 0.7 不降」，方向感是唯一能区分「在学」和「站着不动」的信号
（奖励分本身二义，见 eval_v03_velocity.py 的 docstring）。

箭头取**最近 20% 的点**与**前 20% 的点**的均值对比，避开噪声。

用法::

    uv run python scripts/v03_watch.py            # 最新 run
    uv run python scripts/v03_watch.py --run <目录名>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v03_train_dashboard import load, newest_run  # noqa: E402


def series(acc: EventAccumulator, key: str) -> list[tuple[int, float]]:
  try:
    return [(s.step, s.value) for s in acc.Scalars(key)]
  except KeyError:
    return []


def tail_mean(pts: list[tuple[int, float]], frac: float = 0.2) -> float | None:
  """取最后 frac 比例的点求均值（至少 1 个）。"""
  if not pts:
    return None
  n = max(1, int(len(pts) * frac))
  vals = [v for _, v in pts[-n:]]
  return sum(vals) / len(vals)


def head_mean(pts: list[tuple[int, float]], frac: float = 0.2) -> float | None:
  if not pts:
    return None
  n = max(1, int(len(pts) * frac))
  vals = [v for _, v in pts[:n]]
  return sum(vals) / len(vals)


def arrow(pts: list[tuple[int, float]], lower_is_better: bool = True) -> str:
  """把「起点均值 → 终点均值 + 方向箭头」压成一个字符串。"""
  a, b = head_mean(pts), tail_mean(pts)
  if a is None or b is None:
    return "—"
  if abs(b - a) < 0.01 * max(abs(a), 1e-6):
    mark = "→"  # 基本没动 —— r01 站桩时就是这个形状
  elif (b < a) == lower_is_better:
    mark = "↓" if lower_is_better else "↑"
  else:
    mark = "↑" if lower_is_better else "↓"
  return f"{a:.2f}→{b:.2f} {mark}"


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", default=None)
  args = ap.parse_args()

  run = newest_run(args.run)
  acc = load(run)

  def last_of(key: str) -> float | None:
    s = series(acc, key)
    return s[-1][1] if s else None

  it = None
  for k in ("Train/mean_reward", "Episode_Reward/track_linear_velocity"):
    s = series(acc, k)
    if s:
      it = s[-1][0]
      break

  def g(key: str, fmt: str = "+.3f") -> str:
    v = last_of(key)
    return format(v, fmt) if v is not None else "—"

  err_xy = arrow(series(acc, "Metrics/twist/error_vel_xy"))
  err_yaw = arrow(series(acc, "Metrics/twist/error_vel_yaw"))

  print(
    f"iter={it} ep_len={g('Train/mean_episode_length', '.0f')} "
    f"rew={g('Train/mean_reward', '+.1f')} "
    f"| err_xy {err_xy} | err_yaw {err_yaw} "
    f"| air_time={g('Episode_Reward/air_time')} "
    f"alt={g('Episode_Reward/alternating_air_time')} "
    f"| track_lin={g('Episode_Reward/track_linear_velocity', '.2f')}/2.0 "
    f"| fell={g('Episode_Termination/fell_over', '.2f')}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
