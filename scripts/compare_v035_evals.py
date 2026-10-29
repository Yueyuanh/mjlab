"""横向对比多组 ``eval_v035.py`` 结果（walk v1 / v2 / v3 / v5 / crawl）。

E1 站桩基线不进这张图：它在四档速度上的实测都在 0 附近，四根柱全是 0，
不承载信息；E1 的故事在 §4.1 单独讲。这样做也正好把系列数压到 5 个——
分类色板只有 5 个通过色盲与常规视觉分辨门的色相，第 6 个硬凑会掉到 ΔE 11.7。

每组评估的命令幅值不同（walk 前进 0.20、crawl 0.10，转向两档都是 0.15），
所以第一张图用**达成率 = 实测 / 命令**而不是绝对速度，否则柱高差一半只是
命令本身差了 2 倍。站立档没有命令，只出现在偏航漂移图里。

用法::

  uv run python scripts/compare_v035_evals.py \
    --run "E1=data/biped_e1/eval.json" \
    --run "walk v1=data/walk/eval.json" \
    --run "walk v2=data/walk_v2/eval.json" \
    --run "walk v3=data/walk_v3/eval.json" \
    --run "crawl=data/crawl/eval.json" \
    --output doc/pikachu_v035_20261028/media/figures/fig_compare.png \
    --markdown doc/pikachu_v035_20261028/data/compare.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#d9d8d3"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#a2569f")

# 图中标签是中文，默认 DejaVu Sans 没有汉字字形（渲染成方框），必须显式挂 CJK 字体。
# 本机 Noto Sans CJK 是 NotoSansCJK-Regular.ttc 集合文件，matplotlib 只注册了其中的
# JP 面（字体里没有单独的 "Noto Sans CJK SC"），汉字字形是共用的，用 JP 面即可。
plt.rcParams.update(
  {
    "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
  }
)

# 数值标签压在参考线上会看不清，给个同底色的小背景板。
_LABEL_BG = {"facecolor": SURFACE, "edgecolor": "none", "pad": 0.6}

CLIPS = ("stand", "forward", "backward", "lateral", "turn")
CLIP_ZH = {
  "stand": "站立",
  "forward": "前进",
  "backward": "后退",
  "lateral": "侧移",
  "turn": "转向",
}
# 每个驱动档对应的「被命令轴的实测值」索引：0=x, 1=y, 2=yaw
CLIP_AXIS = {"forward": 0, "backward": 0, "lateral": 1, "turn": 2}
DRIVEN = ("forward", "backward", "lateral", "turn")


def load_runs(specs: list[str]) -> list[tuple[str, dict]]:
  runs = []
  for spec in specs:
    label, _, path = spec.partition("=")
    if not path:
      raise SystemExit(f"--run 需要 label=path 形式，收到 {spec!r}")
    runs.append((label, json.loads(Path(path).read_text())))
  return runs


def achievement(clip: str, entry: dict) -> float:
  axis = CLIP_AXIS[clip]
  cmd = entry["command"][axis]
  actual = entry["mean_velocity"][axis]
  return actual / cmd if abs(cmd) > 1e-9 else float("nan")


def _style(ax, title: str, ylabel: str) -> None:
  ax.set_facecolor(SURFACE)
  ax.set_title(title, color=TEXT, fontsize=10, loc="left", pad=8)
  ax.set_ylabel(ylabel, color=TEXT_MUTED, fontsize=8)
  ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.9)
  ax.set_axisbelow(True)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
  ax.tick_params(colors=TEXT_MUTED, labelsize=8, length=0)


def plot(runs: list[tuple[str, dict]], out: Path) -> None:
  fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(12.4, 4.0), gridspec_kw={"width_ratios": (1.0, 1.25)}
  )
  fig.patch.set_facecolor(SURFACE)

  # --- 左：各驱动档的达成率（理想 = 1.0） ---
  width = 0.8 / len(runs)
  x = np.arange(len(DRIVEN))
  for i, (label, data) in enumerate(runs):
    vals = [achievement(c, data["clips"][c]) for c in DRIVEN]
    bars = ax_a.bar(
      x - 0.4 + width * (i + 0.5),
      vals,
      width * 0.88,
      color=SERIES[i % len(SERIES)],
      label=label,
      linewidth=1.2,
      edgecolor=SURFACE,
    )
    for bar, v in zip(bars, vals, strict=True):
      # 零附近的柱子（例如 E1 完全不动）如果按柱子顶/底放标签，会压到 x 轴刻度上，
      # 这种情况统一把标签抬到零线之上，靠字号和位置区分即可。
      tiny = abs(v) < 0.06
      y = 0.025 if tiny else (v + 0.035 if v > 0 else v - 0.045)
      ax_a.text(
        bar.get_x() + bar.get_width() / 2,
        y,
        f"{v:.0%}",
        ha="center",
        va="bottom" if (tiny or v > 0) else "top",
        fontsize=6.5,
        color=TEXT_MUTED,
        bbox=_LABEL_BG,
      )
  ax_a.axhline(1.0, color=TEXT_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
  ax_a.text(
    len(DRIVEN) - 0.45, 1.02, "命令值", fontsize=7, color=TEXT_MUTED, ha="right"
  )
  ax_a.set_xticks(x, [CLIP_ZH[c] for c in DRIVEN])
  ax_a.set_ylim(-0.15, 1.42)
  _style(ax_a, "速度达成率（实测 / 命令，不同档命令幅值不同）", "达成率")

  # --- 右：零命令轴的偏航漂移（理想 = 0，越小越好） ---
  xb = np.arange(len(CLIPS))
  for i, (label, data) in enumerate(runs):
    vals = [abs(data["clips"][c]["mean_velocity"][2]) for c in CLIPS]
    bars = ax_b.bar(
      xb - 0.4 + width * (i + 0.5),
      vals,
      width * 0.88,
      color=SERIES[i % len(SERIES)],
      label=label,
      linewidth=1.2,
      edgecolor=SURFACE,
    )
    for bar, v in zip(bars, vals, strict=True):
      ax_b.text(
        bar.get_x() + bar.get_width() / 2,
        v + 0.004,
        f"{v:.3f}",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=TEXT_MUTED,
        bbox=_LABEL_BG,
      )
  ax_b.set_xticks(xb, [CLIP_ZH[c] for c in CLIPS])
  top = max(
    0.17,
    max(abs(d["clips"][c]["mean_velocity"][2]) for _, d in runs for c in CLIPS) * 1.3,
  )
  ax_b.set_ylim(0, top)
  # 只有转向档的 wz 命令非零（0.15），画成参考线才能看出「转向到底走了几成」。
  ax_b.axhline(0.15, color=TEXT_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
  ax_b.text(
    len(CLIPS) - 0.45, 0.154, "转向命令 0.15", fontsize=7, color=TEXT_MUTED, ha="right"
  )
  _style(ax_b, "偏航漂移 |wz|（站立/前进/后退/侧移的命令为 0）", "rad/s")

  handles, labels = ax_a.get_legend_handles_labels()
  fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=len(runs),
    frameon=False,
    fontsize=8.5,
    bbox_to_anchor=(0.5, -0.02),
  )
  fig.tight_layout(rect=(0, 0.07, 1, 1))
  out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
  plt.close(fig)


def markdown(runs: list[tuple[str, dict]]) -> str:
  head = ["指标"] + [label for label, _ in runs]
  lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]

  def row(name: str, fn) -> None:
    cells = [name]
    for _, data in runs:
      try:
        cells.append(fn(data))
      except (KeyError, IndexError):
        cells.append("—")
    lines.append("| " + " | ".join(cells) + " |")

  row(
    "前进实测 / 命令 (m/s)",
    lambda d: (
      f"{d['clips']['forward']['mean_velocity'][0]:+.3f} / {d['clips']['forward']['command'][0]:.2f}"
    ),
  )
  row(
    "后退实测 / 命令 (m/s)",
    lambda d: (
      f"{d['clips']['backward']['mean_velocity'][0]:+.3f} / {d['clips']['backward']['command'][0]:.2f}"
    ),
  )
  row(
    "侧移实测 / 命令 (m/s)",
    lambda d: (
      f"{d['clips']['lateral']['mean_velocity'][1]:+.3f} / {d['clips']['lateral']['command'][1]:.2f}"
    ),
  )
  row(
    "转向实测 / 命令 (rad/s)",
    lambda d: (
      f"{d['clips']['turn']['mean_velocity'][2]:+.3f} / {d['clips']['turn']['command'][2]:.2f}"
    ),
  )
  row("站立 wz (rad/s)", lambda d: f"{d['clips']['stand']['mean_velocity'][2]:+.3f}")
  row(
    "前进 wz 漂移 (rad/s)",
    lambda d: f"{d['clips']['forward']['mean_velocity'][2]:+.3f}",
  )
  row("前进 MAE vx", lambda d: f"{d['clips']['forward']['velocity_mae'][0]:.3f}")
  row("转向 MAE wz", lambda d: f"{d['clips']['turn']['velocity_mae'][2]:.3f}")
  row(
    "平均支撑数",
    lambda d: f"{np.mean([d['clips'][c]['mean_support_count'] for c in CLIPS]):.2f}",
  )
  row(
    "摔倒次数（5 档合计）",
    lambda d: str(sum(d["clips"][c]["falls"] for c in CLIPS)),
  )
  return "\n".join(lines)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", action="append", required=True, help="label=evel.json")
  ap.add_argument("--output", type=Path, required=True)
  ap.add_argument("--markdown", type=Path)
  args = ap.parse_args()

  runs = load_runs(args.run)
  plot(runs, args.output)
  md = markdown(runs)
  print(md)
  if args.markdown:
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(md + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
