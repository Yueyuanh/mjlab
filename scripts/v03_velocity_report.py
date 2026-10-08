"""Pikachu v03 velocity 任务报告生成器：metrics CSV → HTML → PDF。

## 输入

  - ``scripts/record_v03_metrics.py`` 写出的逐档 CSV（``metrics_v03_<档>_<tag>.csv``）；
  - 可选：tensorboard 训练日志目录（画 r04/r05/r06 的曲线对比）；
  - 可选：``--notes`` 指定的 Markdown 分析正文，插到报告的「分析」一节。

## 输出

  - 自包含 HTML（图表是 base64 内联 PNG，**零外部依赖、零网络**）；
  - PDF（headless Chrome ``--print-to-pdf``；本机无 weasyprint/wkhtmltopdf，
    Chrome 是唯一装好的 HTML→PDF 通路）。

## 两个刻意的取舍

1. **不用 pandas**：本机没装。列数 80+、行数几百，纯 stdlib 的 ``csv`` +
   ``numpy`` 足够，少一个重依赖。
2. **tensorboard 用流式读**（``EventFileLoader``）而不是 ``EventAccumulator``。
   r04 有 116 个 tag × 30000 点，Accumulator 全量 Reload 实测 **73 秒**；
   而报告只要 5~6 个 tag，流式过滤能省掉绝大部分。曲线还会缓存成 JSON，
   重出报告不必重读。

用法::

    cd mjlab
    uv run python scripts/v03_velocity_report.py \
        --metrics-dir docs/pikachu_v03/metrics --tag r06-best \
        --checkpoint logs/rsl_rl/pikachu_v03_velocity/<run>/model_54999.pt \
        --out docs/pikachu_v03/reports/velocity-report.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 中文字体：本机装了 Noto Sans CJK JP（覆盖汉字），回退 Droid Sans Fallback。
plt.rcParams["font.sans-serif"] = [
  "Noto Sans CJK JP",
  "Droid Sans Fallback",
  "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

# 档位顺序（报告里的行序）与中文标签。
CLIP_ORDER = ["stand", "fwd", "fwd_fast", "back", "lat", "turn"]
CLIP_LABEL = {
  "stand": "站立",
  "fwd": "前进 0.40",
  "fwd_fast": "前进 0.50",
  "back": "后退 0.30",
  "lat": "侧移 0.25",
  "turn": "转向 0.30",
}

# 报告要对比的三轮（tensorboard 目录名正则 -> 显示名）。
RUN_PATTERNS = [
  ("r04", r".*_v03-flat-r04$", "r04（无站姿项，跑满 30k）"),
  ("r05", r".*_v03-flat-r05$", "r05（站姿项无门控）"),
  ("r06", r".*_v03-flat-r06$", "r06（站姿项带门控）"),
]

CURVE_TAGS = [
  ("Episode_Reward/track_linear_velocity", "线速度追踪奖励"),
  ("Episode_Reward/track_angular_velocity", "角速度追踪奖励"),
  ("Metrics/twist/error_vel_xy", "线速度误差 ‖Δv_xy‖"),
  ("Metrics/twist/error_vel_yaw", "偏航角速度误差"),
  ("Train/mean_episode_length", "平均回合长度"),
  ("Metrics/foot_separation", "站立时双脚间距 (m)"),
]


# --------------------------------------------------------------------------
# CSV 读取与逐档分析
# --------------------------------------------------------------------------


def load_clip(path: Path) -> dict[str, np.ndarray]:
  """读一个档的 CSV，返回 {列名: float 数组}（元数据列也一并读出来）。"""
  with path.open(encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    cols: dict[str, list[float]] = {name: [] for name in (reader.fieldnames or [])}
    for row in reader:
      for name in cols:
        raw = row.get(name, "")
        try:
          cols[name].append(float(raw))
        except (TypeError, ValueError):
          cols[name].append(float("nan"))
  return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def unwrap_deg(rad: np.ndarray) -> np.ndarray:
  """把 ±π 跳变的偏航角解成连续角（度）。"""
  if rad.size == 0:
    return rad
  d = np.diff(rad)
  d = np.arctan2(np.sin(d), np.cos(d))  # 归一化到 (-π, π]
  return np.concatenate([[0.0], np.cumsum(d)]) * 180.0 / math.pi


def tail_mean(x: np.ndarray, skip_frac: float = 0.125, skip_max: int = 25) -> float:
  """丢掉起步瞬态后取均值。

  起步瞬态是「从 reset 姿态收敛到步态」的那一段，把它算进追踪统计会系统性
  低估 —— 它不是稳态性能，是收敛过程。与 ``record_v03_metrics.summarize``
  同一口径，保证终端数字和报告数字一致。
  """
  if x.size == 0:
    return float("nan")
  skip = min(int(x.size * skip_frac), skip_max)
  body = x[skip:]
  return float(np.nanmean(body)) if body.size else float("nan")


def analyze_clip(clip: str, c: dict[str, np.ndarray]) -> dict:
  """算一个档的追踪指标（全部来自物理量，不看奖励）。"""
  has_vel = "vel/cmd_x" in c and c["vel/cmd_x"].size > 0
  t = c.get("time_s", np.asarray([]))
  out: dict = {
    "clip": clip,
    "label": CLIP_LABEL.get(clip, clip),
    "seconds": float(t[-1] - t[0]) if t.size > 1 else 0.0,
    "samples": int(t.size),
  }

  if has_vel:
    act = np.array([tail_mean(c["vel/act_x"]), tail_mean(c["vel/act_y"]),
                    tail_mean(c["vel/act_yaw"])])
    # 命令是常量（ranges 收成一个点），取整段均值即可，且能反过来验证「真的固定住了」。
    cmd_set = np.array([np.nanmean(c["vel/cmd_x"]), np.nanmean(c["vel/cmd_y"]),
                        np.nanmean(c["vel/cmd_yaw"])])
    cmd_std = np.array([np.nanstd(c["vel/cmd_x"]), np.nanstd(c["vel/cmd_y"]),
                        np.nanstd(c["vel/cmd_yaw"])])
    cmd_mag = float(np.hypot(cmd_set[0], cmd_set[1]))
    act_mag = float(np.hypot(act[0], act[1]))
    out.update(
      cmd=cmd_set.tolist(),
      cmd_std=cmd_std.tolist(),
      act=act.tolist(),
      # 跟踪比 = 实测速度模 / 命令速度模。命令为 0 时无意义（站立档）。
      track_ratio=act_mag / cmd_mag if cmd_mag > 1e-6 else float("nan"),
      # 方向余弦：+1 走对、-1 反向、0 垂直。只对平移档有意义。
      dir_cos=float(np.dot(act[:2], cmd_set[:2]) / (act_mag * cmd_mag))
      if cmd_mag > 1e-6 and act_mag > 1e-6
      else float("nan"),
      err_xy=tail_mean(c["vel/err_xy_norm"]) if "vel/err_xy_norm" in c else float("nan"),
      err_yaw=tail_mean(np.abs(c["vel/err_yaw"])) if "vel/err_yaw" in c else float("nan"),
      standing_frac=float(np.nanmean(c["vel/standing"])) if "vel/standing" in c else float("nan"),
    )
    if "vel/pos_x_w" in c:
      px, py = c["vel/pos_x_w"], c["vel/pos_y_w"]
      out["net_disp_m"] = float(np.hypot(px[-1] - px[0], py[-1] - py[0]))
    if "vel/yaw_w" in c:
      out["net_yaw_deg"] = float(unwrap_deg(c["vel/yaw_w"])[-1])

  if "foot_lateral_separation_m" in c:
    sep = c["foot_lateral_separation_m"] * 100.0
    out["sep_mean_cm"] = tail_mean(sep)
    out["sep_p95_cm"] = float(np.nanpercentile(sep, 95))

  if "base_height_m" in c:
    out["base_height_cm"] = tail_mean(c["base_height_m"]) * 100.0

  return out


# --------------------------------------------------------------------------
# 逐关节分析
# --------------------------------------------------------------------------


def analyze_joints(clips: dict[str, dict[str, np.ndarray]]) -> list[dict]:
  """把各档的关节列汇总成一张表（峰值取所有档的最大值）。

  峰值口径取「全档最大」而不是某档：报告要回答的是**电机够不够用**，
  那就该看整段录制里最恶劣的那一帧。
  """
  joints: list[str] = []
  for c in clips.values():
    for name in c:
      if name.startswith("joint_position_rad/"):
        j = name[len("joint_position_rad/") :]
        if j not in joints:
          joints.append(j)
  joints.sort(key=lambda j: (0 if j.startswith("left") else 1, j))

  rows = []
  for j in joints:
    pos_lo = pos_hi = float("nan")
    peak_v = peak_t = 0.0
    peak_vr = peak_tr = 0.0
    peak_p = peak_a = 0.0
    mean_abs_t: list[float] = []
    mean_abs_p: list[float] = []
    for c in clips.values():
      p = c.get(f"joint_position_rad/{j}")
      if p is not None and p.size:
        lo, hi = float(np.nanmin(p)), float(np.nanmax(p))
        pos_lo = lo if math.isnan(pos_lo) else min(pos_lo, lo)
        pos_hi = hi if math.isnan(pos_hi) else max(pos_hi, hi)
      for key, acc in ((f"joint_velocity_rad_s/{j}", "v"), (f"joint_torque_nm/{j}", "t")):
        x = c.get(key)
        if x is not None and x.size:
          peak = float(np.nanmax(np.abs(x)))
          if acc == "v":
            peak_v = max(peak_v, peak)
          else:
            peak_t = max(peak_t, peak)
            mean_abs_t.append(float(np.nanmean(np.abs(x))))
      x = c.get(f"joint_velocity_ratio/{j}")
      if x is not None and x.size:
        peak_vr = max(peak_vr, float(np.nanmax(np.abs(x))))
      x = c.get(f"joint_torque_ratio/{j}")
      if x is not None and x.size:
        peak_tr = max(peak_tr, float(np.nanmax(np.abs(x))))
      # 机械功率：峰值是选电池/选驱动的依据，均值反映发热。
      x = c.get(f"joint_power_w/{j}")
      if x is not None and x.size:
        peak_p = max(peak_p, float(np.nanmax(np.abs(x))))
        mean_abs_p.append(float(np.nanmean(np.abs(x))))
      x = c.get(f"joint_accel_rad_s2/{j}")
      if x is not None and x.size:
        peak_a = max(peak_a, float(np.nanmax(np.abs(x))))

    # 从占用比反推电机限值：ratio = 实测 / 限值，两个口径应当自洽。
    v_limit = peak_v / peak_vr if peak_vr > 1e-9 else float("nan")
    t_limit = peak_t / peak_tr if peak_tr > 1e-9 else float("nan")
    rows.append(
      dict(
        joint=j,
        side="左" if j.startswith("left") else "右",
        short=j.replace("_joint", "").replace("left_", "").replace("right_", ""),
        pos_lo_rad=pos_lo,
        pos_hi_rad=pos_hi,
        rom_deg=(pos_hi - pos_lo) * 180.0 / math.pi if not math.isnan(pos_lo) else float("nan"),
        peak_v=peak_v,
        peak_v_ratio=peak_vr,
        v_limit=v_limit,
        peak_t=peak_t,
        peak_t_ratio=peak_tr,
        t_limit=t_limit,
        mean_abs_t=float(np.mean(mean_abs_t)) if mean_abs_t else float("nan"),
        peak_p=peak_p if mean_abs_p else float("nan"),
        mean_abs_p=float(np.mean(mean_abs_p)) if mean_abs_p else float("nan"),
        peak_a=peak_a,
      )
    )
  return rows


# --------------------------------------------------------------------------
# 图表（matplotlib -> base64 内联 PNG）
# --------------------------------------------------------------------------


def fig_uri(fig) -> str:
  buf = io.BytesIO()
  fig.savefig(buf, format="png")
  plt.close(fig)
  return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def chart_tracking(clip: str, c: dict[str, np.ndarray]) -> str | None:
  """单档三面板：vx / vy / wz 的命令 vs 实测。"""
  if "vel/cmd_x" not in c:
    return None
  t = c["time_s"]
  fig, axes = plt.subplots(1, 3, figsize=(11.5, 2.7))
  for ax, (axis, unit) in zip(
    axes, (("x", "m/s"), ("y", "m/s"), ("yaw", "rad/s")), strict=True
  ):
    cmd, act = c[f"vel/cmd_{axis}"], c[f"vel/act_{axis}"]
    ax.plot(t, cmd, lw=1.6, ls="--", color="#6c757d", label="命令")
    ax.plot(t, act, lw=1.4, color="#0d6efd", label="实测")
    ax.set_title(f"v{axis} ({unit})", fontsize=9)
    ax.set_xlabel("t (s)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)
  axes[0].legend(fontsize=7, loc="best")
  fig.suptitle(f"{CLIP_LABEL.get(clip, clip)} —— 速度追踪", fontsize=10, y=1.06)
  return fig_uri(fig)


def chart_separation(clips: dict[str, dict[str, np.ndarray]]) -> str | None:
  """六档的双脚横向间距时序（2×3）。"""
  have = [k for k in CLIP_ORDER if k in clips and "foot_lateral_separation_m" in clips[k]]
  if not have:
    return None
  fig, axes = plt.subplots(2, 3, figsize=(11.5, 4.4), sharey=True)
  for ax, clip in zip(axes.ravel(), have, strict=False):
    c = clips[clip]
    ax.plot(c["time_s"], c["foot_lateral_separation_m"] * 100.0, lw=1.3, color="#198754")
    ax.axhline(20.0, color="#dc3545", lw=1.0, ls=":", label="默认站姿 20 cm")
    ax.set_title(CLIP_LABEL.get(clip, clip), fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=7)
    ax.set_ylim(0, 26)
  for ax in axes.ravel()[len(have) :]:
    ax.axis("off")
  axes[0, 0].legend(fontsize=7)
  for ax in axes[-1]:
    ax.set_xlabel("t (s)", fontsize=8)
  for ax in axes[:, 0]:
    ax.set_ylabel("双脚间距 (cm)", fontsize=8)
  fig.suptitle("双脚横向间距（base 朝向系 |Δy|）—— 走路档应当接近 0（迈步时并拢）",
               fontsize=10, y=1.02)
  return fig_uri(fig)


def chart_joints(jrows: list[dict]) -> str:
  """关节 ROM + 峰值占用率 + 机械功率三张横条图。"""
  names = [f"{r['side']}·{r['short']}" for r in jrows]
  y = np.arange(len(jrows))
  fig, axes = plt.subplots(1, 4, figsize=(13.0, 4.2))

  ax = axes[0]
  lo = [r["pos_lo_rad"] * 180 / math.pi for r in jrows]
  hi = [r["pos_hi_rad"] * 180 / math.pi for r in jrows]
  ax.barh(y, [h - lo_i for h, lo_i in zip(hi, lo, strict=True)], left=lo,
          color="#0d6efd", alpha=0.8)
  ax.set_yticks(y, names, fontsize=7)
  ax.set_xlabel("关节角范围 (°)", fontsize=8)
  ax.set_title("活动范围 ROM", fontsize=9)
  ax.grid(alpha=0.25, lw=0.5, axis="x")
  ax.tick_params(labelsize=7)

  ax = axes[1]
  ax.barh(y, [r["peak_t_ratio"] for r in jrows], color="#fd7e14", alpha=0.85)
  ax.axvline(1.0, color="#dc3545", lw=1.2, ls="--", label="电机额定 1.0")
  ax.set_yticks(y, ["" for _ in jrows])
  ax.set_xlabel("峰值扭矩 / 额定", fontsize=8)
  ax.set_title("扭矩占用率", fontsize=9)
  ax.grid(alpha=0.25, lw=0.5, axis="x")
  ax.legend(fontsize=7)
  ax.tick_params(labelsize=7)

  ax = axes[2]
  ax.barh(y, [r["peak_v_ratio"] for r in jrows], color="#6f42c1", alpha=0.85)
  ax.axvline(1.0, color="#dc3545", lw=1.2, ls="--")
  ax.set_yticks(y, ["" for _ in jrows])
  ax.set_xlabel("峰值角速度 / 额定", fontsize=8)
  ax.set_title("速度占用率", fontsize=9)
  ax.grid(alpha=0.25, lw=0.5, axis="x")
  ax.tick_params(labelsize=7)

  # 机械功率：选电池/选驱动的依据。峰值（实心）与均值（描边）并排。
  ax = axes[3]
  ax.barh(y + 0.18, [r["peak_p"] for r in jrows], height=0.34,
          color="#198754", alpha=0.9, label="峰值")
  ax.barh(y - 0.18, [r["mean_abs_p"] for r in jrows], height=0.34,
          color="#198754", alpha=0.35, label="均值")
  ax.set_yticks(y, ["" for _ in jrows])
  ax.set_xlabel("|机械功率| (W)", fontsize=8)
  ax.set_title("关节功率", fontsize=9)
  ax.grid(alpha=0.25, lw=0.5, axis="x")
  ax.legend(fontsize=7)
  ax.tick_params(labelsize=7)

  fig.suptitle("逐关节负载（所有档位的最大值）", fontsize=10, y=1.02)
  return fig_uri(fig)


def chart_curves(curves: dict[str, dict[str, list]]) -> str | None:
  """r04/r05/r06 的训练曲线对比（2×3）。"""
  tags = [t for t, _ in CURVE_TAGS if any(t in v for v in curves.values())]
  if not tags:
    return None
  n = len(tags)
  fig, axes = plt.subplots(2, 3, figsize=(11.5, 4.6))
  colors = {"r04": "#0d6efd", "r05": "#fd7e14", "r06": "#198754"}
  for ax, tag in zip(axes.ravel(), tags, strict=False):
    for run, series in curves.items():
      if tag not in series or not series[tag]:
        continue
      pts = series[tag]
      step = np.asarray([p[0] for p in pts])
      val = np.asarray([p[1] for p in pts])
      # 抽稀到 ~400 点：30000 点画进 3 英寸宽的图里没有额外信息，只是慢。
      if step.size > 400:
        idx = np.linspace(0, step.size - 1, 400).astype(int)
        step, val = step[idx], val[idx]
      ax.plot(step, val, lw=1.2, color=colors.get(run, "#666"), label=run)
    label = dict(CURVE_TAGS)[tag]
    ax.set_title(label, fontsize=9)
    ax.set_xlabel("迭代", fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=7)
  for ax in axes.ravel()[n:]:
    ax.axis("off")
  axes[0, 0].legend(fontsize=7)
  fig.suptitle("训练曲线对比（r04 无站姿项 / r05 无门控 / r06 带门控）", fontsize=10, y=1.02)
  return fig_uri(fig)


# --------------------------------------------------------------------------
# tensorboard 流式读取
# --------------------------------------------------------------------------


def read_curves(log_root: Path, cache: Path) -> dict[str, dict[str, list]]:
  """读三轮的指定 tag，结果缓存成 JSON（重出报告不必重读）。

  ⚠️ **实测过两条路，最后选了慢的那条**：

  - ``EventFileLoader`` 流式过滤（本想只留 6 个 tag 省内存）：**122 秒**，
    而且**取不到值** —— mjlab 写的是 tensor proto，``v.HasField("simple_value")``
    恒为 False（事件总数 348 万，逐个 Python 迭代本身就慢）。
  - ``EventAccumulator`` 全量 Reload：**73 秒**，值正确（它会同时处理
    ``simple_value`` 和 ``tensor`` 两种编码）。

  所以用 Accumulator，代价靠 **JSON 缓存**摊掉 —— 第二次出报告是毫秒级。
  三个 run 一次性约 4 分钟，且只在训练停下之后跑。
  """
  if cache.exists():
    print(f"  [curves] 用缓存 {cache}")
    return json.loads(cache.read_text(encoding="utf-8"))

  import re
  import time

  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  want = {t for t, _ in CURVE_TAGS}
  out: dict[str, dict[str, list]] = {}
  for run, pattern, _label in RUN_PATTERNS:
    dirs = [d for d in sorted(log_root.glob("*")) if d.is_dir() and re.match(pattern, d.name)]
    if not dirs:
      print(f"  [curves] 跳过 {run}：没有匹配 {pattern} 的目录", file=sys.stderr)
      continue
    evs = sorted(dirs[-1].glob("events.out.tfevents.*"))
    if not evs:
      continue
    t0 = time.time()
    acc = EventAccumulator(str(evs[0]), size_guidance={"scalars": 0})
    acc.Reload()
    present = set(acc.Tags()["scalars"])
    series: dict[str, list] = {}
    for tag in want & present:
      series[tag] = [[s.step, s.value] for s in acc.Scalars(tag)]
    out[run] = series
    print(f"  [curves] {run}: {dirs[-1].name} -> " +
          ", ".join(f"{t.split('/')[-1]}={len(v)}" for t, v in series.items() if v) +
          f"  ({time.time() - t0:.0f}s)")
  cache.parent.mkdir(parents=True, exist_ok=True)
  cache.write_text(json.dumps(out), encoding="utf-8")
  return out


# --------------------------------------------------------------------------
# 参数快照
# --------------------------------------------------------------------------


def collect_params(task: str) -> dict:
  """从任务配置里读出报告要展示的参数（命令空间 / 奖励 / 课程）。"""
  import mjlab.tasks  # noqa: F401
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(task, play=False)
  twist = cfg.commands["twist"]
  r = twist.ranges

  def rng(x) -> str:
    return f"[{x[0]:+.2f}, {x[1]:+.2f}]" if x else "—"

  params: dict = {
    "task": task,
    "num_envs": cfg.scene.num_envs,
    "episode_length_s": cfg.episode_length_s,
    "step_dt": cfg.decimation * cfg.sim.mujoco.timestep,
    "decimation": cfg.decimation,
    "sim_dt": cfg.sim.mujoco.timestep,
    "command": {
      "lin_vel_x": rng(r.lin_vel_x),
      "lin_vel_y": rng(r.lin_vel_y),
      "ang_vel_z": rng(r.ang_vel_z),
      "heading": rng(r.heading),
      "heading_command": twist.heading_command,
      "heading_control_stiffness": twist.heading_control_stiffness,
      "resampling_time_range": list(twist.resampling_time_range),
      "rel_standing_envs": twist.rel_standing_envs,
      "rel_heading_envs": twist.rel_heading_envs,
      "rel_world_envs": twist.rel_world_envs,
      "rel_forward_envs": getattr(twist, "rel_forward_envs", None),
      "rel_backward_envs": getattr(twist, "rel_backward_envs", None),
      "rel_lateral_envs": getattr(twist, "rel_lateral_envs", None),
    },
    # 课程项的参数结构随实现而变（``commands_vel`` 用的是 params 里的
    # ``velocity_stages`` 列表，不是 ``CurriculumTermCfg.stages`` 属性 ——
    # 后者根本不存在）。所以按「参数名 -> 格式化后的 params」通用地导出，
    # 不假设任何具体字段。
    "curriculum": {name: _fmt_param(term.params) for name, term in cfg.curriculum.items()},
    "rewards": [],
    "metrics_count": len(cfg.metrics),
    "joints": sorted(
      {
        n.split("/", 1)[1]
        for n in cfg.metrics
        if n.startswith("joint_position_rad/")
      }
    ),
  }

  for name, term in cfg.rewards.items():
    fn = getattr(term.func, "__name__", str(term.func))
    ps = {k: _fmt_param(v) for k, v in (term.params or {}).items()}
    params["rewards"].append({"name": name, "func": fn, "weight": term.weight, "params": ps})
  return params


def _fmt_param(v) -> str:
  if isinstance(v, (int, float, str, bool)) or v is None:
    return repr(v)
  if isinstance(v, (tuple, list)):
    return "[" + ", ".join(_fmt_param(x) for x in v) + "]"
  if isinstance(v, dict):
    return "{" + ", ".join(f"{k}: {_fmt_param(x)}" for k, x in v.items()) + "}"
  return type(v).__name__


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

_CSS = """
:root{--fg:#1f2328;--mut:#59636e;--line:#d1d9e0;--bg:#fff;--acc:#0d6efd;
      --ok:#1a7f37;--warn:#9a6700;--bad:#cf222e;--soft:#f6f8fa}
*{box-sizing:border-box}
body{margin:0;padding:28px 34px;background:var(--bg);color:var(--fg);
     font:14px/1.65 "Noto Sans CJK JP","Droid Sans Fallback",system-ui,sans-serif}
h1{font-size:24px;margin:0 0 4px;border-bottom:3px solid var(--acc);padding-bottom:8px}
h2{font-size:18px;margin:30px 0 10px;padding-left:9px;border-left:4px solid var(--acc)}
h3{font-size:15px;margin:20px 0 8px;color:var(--mut)}
p,li{color:var(--fg)}
.sub{color:var(--mut);font-size:13px;margin:0 0 18px}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:11.5px}
th,td{border:1px solid var(--line);padding:5px 7px;text-align:right}
/* nowrap 是给 A4 竖版用的：宽表在 12mm 页边距下会被挤成两行，
   「跟踪比」裂成「跟踪 / 比」很难读。宁可让列自己撑开。 */
th{background:var(--soft);font-weight:600;text-align:center;white-space:nowrap}
td:first-child,th:first-child{text-align:left;white-space:nowrap}
/* 宽表（10+ 列）在 A4 竖版里放不下：单元格会折成三行，反而更难读。
   给它们单独一档更小的字号 + 全部 nowrap，让列自己撑开。 */
table.wide{font-size:10.5px}
table.wide th,table.wide td{padding:4px 5px;white-space:nowrap}
tr:nth-child(even) td{background:#fbfcfd}
code{background:var(--soft);padding:1px 5px;border-radius:3px;
     font-family:"Noto Sans Mono",ui-monospace,monospace;font-size:12px}
pre{background:var(--soft);border:1px solid var(--line);border-radius:6px;
    padding:10px 12px;overflow-x:auto;font-size:11.5px;line-height:1.5}
.cards{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}
.card{border:1px solid var(--line);border-radius:8px;padding:9px 14px;min-width:132px}
.card .k{font-size:11px;color:var(--mut);letter-spacing:.04em}
.card .v{font-size:19px;font-weight:600;margin-top:2px}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
figure{margin:14px 0;page-break-inside:avoid}
figure img{width:100%;border:1px solid var(--line);border-radius:6px}
figcaption{font-size:12px;color:var(--mut);margin-top:5px}
.note{background:#fff8e6;border-left:4px solid #d4a72c;padding:9px 13px;
      border-radius:0 6px 6px 0;margin:12px 0;font-size:13px}
.good{background:#eaf7ee;border-left-color:var(--ok)}
section{page-break-inside:auto}
h2{page-break-after:avoid}
@media print{body{padding:0}@page{size:A4;margin:12mm}}
"""


def esc(x) -> str:
  return (
    str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
  )


def fmt(x, nd=3, dash="—") -> str:
  if x is None:
    return dash
  try:
    xf = float(x)
  except (TypeError, ValueError):
    return esc(x)
  if math.isnan(xf):
    return dash
  return f"{xf:.{nd}f}"


def cls_for_ratio(r: float, good: float = 0.8) -> str:
  if r is None or (isinstance(r, float) and math.isnan(r)):
    return ""
  return "ok" if r <= good else ("warn" if r <= 1.0 else "bad")


def table(
  headers: list[str],
  rows: list[list[str]],
  cls: list[str] | None = None,
  table_class: str = "",
) -> str:
  out = [f"<table class='{table_class}'><thead><tr>" if table_class else "<table><thead><tr>"]
  out += [f"<th>{esc(h)}</th>" for h in headers]
  out.append("</tr></thead><tbody>")
  for row in rows:
    out.append("<tr>")
    for i, cell in enumerate(row):
      c = cls[i] if cls and i < len(cls) else ""
      out.append(f'<td class="{c}">{cell}</td>')
    out.append("</tr>")
  out.append("</tbody></table>")
  return "".join(out)


def build_html(
  *,
  tag: str,
  checkpoint: str,
  clips: dict[str, dict[str, np.ndarray]],
  per_clip: list[dict],
  jrows: list[dict],
  params: dict,
  curves_html: str,
  notes_html: str,
  charts: dict[str, str],
  generated: str,
) -> str:
  P: list[str] = []
  P.append("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>")
  P.append("<title>Pikachu v03 速度追踪报告</title>")
  P.append(f"<style>{_CSS}</style></head><body>")

  P.append("<h1>Pikachu v03 行走策略 —— 速度追踪与关节负载报告</h1>")
  P.append(
    f"<p class='sub'>检查点 <code>{esc(tag)}</code> · "
    f"<code>{esc(checkpoint)}</code><br>生成于 {esc(generated)} · "
    f"数据来源 <code>scripts/record_v03_metrics.py</code> 录制的逐档 metrics CSV</p>"
  )

  # ---- 概要卡片 ----
  total_samples = sum(c["samples"] for c in per_clip)
  n_term = sum(1 for c in per_clip if c.get("terminated_at"))
  worst_t = max((r["peak_t_ratio"] for r in jrows), default=float("nan"))
  worst_v = max((r["peak_v_ratio"] for r in jrows), default=float("nan"))
  stand = next((c for c in per_clip if c["clip"] == "stand"), None)
  P.append("<div class='cards'>")
  P.append(f"<div class='card'><div class='k'>录制档位</div><div class='v'>{len(per_clip)}</div></div>")
  P.append(f"<div class='card'><div class='k'>采样点</div><div class='v'>{total_samples:,}</div></div>")
  P.append(f"<div class='card'><div class='k'>关节数</div><div class='v'>{len(jrows)}</div></div>")
  P.append(
    f"<div class='card'><div class='k'>中途终止</div>"
    f"<div class='v {'ok' if n_term == 0 else 'bad'}'>{n_term} 档</div></div>"
  )
  P.append(
    f"<div class='card'><div class='k'>峰值扭矩占用</div>"
    f"<div class='v {cls_for_ratio(worst_t)}'>{fmt(worst_t, 2)}×</div></div>"
  )
  P.append(
    f"<div class='card'><div class='k'>峰值速度占用</div>"
    f"<div class='v {cls_for_ratio(worst_v)}'>{fmt(worst_v, 2)}×</div></div>"
  )
  if stand and not math.isnan(stand.get("sep_mean_cm", float("nan"))):
    P.append(
      f"<div class='card'><div class='k'>站立双脚间距</div>"
      f"<div class='v'>{fmt(stand['sep_mean_cm'], 1)} cm</div></div>"
    )
  P.append("</div>")

  # ---- §1 速度追踪总表 ----
  P.append("<h2>1. 速度追踪总表</h2>")
  P.append(
    "<p>命令是<strong>在构造环境之前把 <code>twist.ranges</code> 收成一个点</strong>固定的"
    "（不是事后改张量），所以每个档位全程只有一条命令。"
    "「实测」是 <strong>base 系速度的时均</strong>（丢掉起步瞬态），"
    "与命令同坐标系，可直接相减。"
    "<strong>净位移是世界系首尾差</strong> —— 它才是「到底走没走」的判据；"
    "而 <code>Episode_Reward/track_*</code> 因为 Jensen 不等式站桩也能拿高分，不能当进度指标。</p>"
  )
  rows = []
  for c in per_clip:
    if "cmd" not in c:
      continue
    cmd, act = c["cmd"], c["act"]
    ratio = c["track_ratio"]
    rc = "ok" if (not math.isnan(ratio) and 0.8 <= ratio <= 1.2) else (
      "warn" if (not math.isnan(ratio) and 0.5 <= ratio) else "bad"
    )
    # 被终止的档要标出来：它的均值是「摔之前的平均」，不是稳态性能。
    term = c.get("terminated_at")
    label = esc(c["label"]) + (" <span class='bad'>⚠️</span>" if term else "")
    rows.append([
      label,
      f"{c.get('seconds', float('nan')):.1f}"
      + (f"<span class='bad'>/{fmt(c.get('requested_seconds'), 0)}</span>"
         if term and c.get("requested_seconds") else ""),
      f"{cmd[0]:+.2f}/{cmd[1]:+.2f}/{cmd[2]:+.2f}",
      f"{act[0]:+.2f}/{act[1]:+.2f}/{act[2]:+.2f}",
      f"<span class='{rc}'>{fmt(ratio, 2)}</span>" if not math.isnan(ratio) else "—",
      fmt(c.get("dir_cos"), 2),
      fmt(c.get("err_xy"), 3),
      fmt(c.get("err_yaw"), 3),
      fmt(c.get("net_disp_m"), 2),
      fmt(c.get("net_yaw_deg"), 1),
      fmt(c.get("base_height_cm"), 1),
    ])
  P.append(table(
    ["档位", "时长 (s)", "命令 vx/vy/wz", "实测 vx/vy/wz", "跟踪比", "方向余弦",
     "线速度误差", "偏航误差", "净位移 (m)", "净转角 (°)", "基座高度 (cm)"],
    rows,
    table_class="wide",
  ))
  P.append(
    "<p class='sub'>跟踪比 = ‖实测 v<sub>xy</sub>‖ / ‖命令 v<sub>xy</sub>‖，"
    "命令为 0 的站立档无意义；方向余弦 +1 = 走对方向、−1 = 反向、0 = 垂直。"
    "净转角是解卷绕后的世界系偏航变化 —— 它同时是「净位移能不能读作前进距离」的前提"
    "（转身越大，世界位移方向偏离 base 前向越多）。</p>"
  )

  # ---- §2 速度时序 ----
  P.append("<h2>2. 速度追踪时序</h2>")
  P.append(
    "<p>虚线是命令、实线是实测。稳态段贴合说明跟得上；"
    "周期性起伏是步态本身的节奏（支撑相/摆动相切换），"
    "<strong>单帧值没有意义，要看均值</strong>。</p>"
  )
  for clip in CLIP_ORDER:
    uri = charts.get(f"track_{clip}")
    if uri:
      P.append(
        f"<figure><img src='{uri}' alt='{esc(clip)} 速度追踪'>"
        f"<figcaption>{esc(CLIP_LABEL.get(clip, clip))} 档 —— "
        f"命令 vs 实测（base 系）</figcaption></figure>"
      )

  # ---- §3 站姿开度 ----
  P.append("<h2>3. 双脚横向间距（站姿开度的门控验证）</h2>")
  P.append(
    "<p>这一节是 r05/r06 那个自变量的证据。间距量的是 "
    "<strong>base 朝向系下的 |Δy|</strong>，不是世界系 y 差 —— "
    "后者会随转身按 cos(yaw) 缩水，机器人侧身时读数趋近 0，那是假象。</p>"
  )
  sep_rows = []
  for c in per_clip:
    if "sep_mean_cm" not in c:
      continue
    sep_rows.append([
      esc(c["label"]),
      fmt(c.get("sep_mean_cm"), 1),
      fmt(c.get("sep_p95_cm"), 1),
      fmt(c.get("standing_frac"), 2),
    ])
  P.append(table(["档位", "平均间距 (cm)", "95 分位 (cm)", "门控命中比例"], sep_rows))
  if charts.get("separation"):
    P.append(f"<figure><img src='{charts['separation']}' alt='双脚间距'>"
             f"<figcaption>六档的双脚横向间距时序（红线 = 默认站姿 20 cm）</figcaption></figure>")

  # ---- §4 关节 ----
  P.append("<h2>4. 逐关节负载</h2>")
  P.append(
    "<p>峰值取<strong>所有档位的最大值</strong>（报告要回答的是「电机够不够用」，"
    "所以看最恶劣的那一帧）。占用率 = 实测 / 电机额定，"
    "<strong>&gt; 1.0 表示该帧超规格</strong>。"
    "额定值（速度 / 扭矩）是常量，写在 "
    "<a href='../03-task-design.md'>03-task-design.md §2</a> 的电机分组表里，"
    "表内不再重复，只由「峰值 ÷ 占用率」自洽校验。</p>"
    "<p>拆成两张表是为了排得下：13 列的表在 A4 竖版里会裂行。</p>"
  )
  kin_rows = []
  load_rows = []
  for r in jrows:
    label = f"{r['side']} {esc(r['short'])}"
    kin_rows.append([
      label,
      f"{r['pos_lo_rad']:+.3f} … {r['pos_hi_rad']:+.3f}",
      fmt(r["rom_deg"], 1),
      fmt(r["peak_v"], 2),
      f"<span class='{cls_for_ratio(r['peak_v_ratio'])}'>{fmt(r['peak_v_ratio'], 2)}</span>",
      fmt(r["peak_a"], 1),
    ])
    load_rows.append([
      label,
      fmt(r["peak_t"], 2),
      f"<span class='{cls_for_ratio(r['peak_t_ratio'])}'>{fmt(r['peak_t_ratio'], 2)}</span>",
      fmt(r["mean_abs_t"], 2),
      fmt(r["peak_p"], 2),
      fmt(r["mean_abs_p"], 2),
    ])
  P.append("<h3>4.1 运动学</h3>")
  P.append(table(
    ["关节", "位置范围 (rad)", "ROM (°)", "峰值速度 (rad/s)", "速度占用", "峰值加速度 (rad/s²)"],
    kin_rows,
  ))
  P.append("<h3>4.2 负载与功率</h3>")
  P.append(table(
    ["关节", "峰值扭矩 (N·m)", "扭矩占用", "均|扭矩| (N·m)", "峰值功率 (W)", "均|功率| (W)"],
    load_rows,
  ))
  if charts.get("joints"):
    P.append(f"<figure><img src='{charts['joints']}' alt='关节负载'>"
             f"<figcaption>逐关节活动范围、电机占用率与机械功率</figcaption></figure>")

  # ---- §5 训练曲线 ----
  P.append("<h2>5. 训练曲线对比</h2>")
  P.append(
    "<p>r04 是没有站姿项的基线、r05 加了站姿项但<strong>没有门控</strong>、"
    "r06 是同一项<strong>加了命令门控</strong>并从 r04 热启动。"
    "要看的正是「加了门控之后，线速度追踪有没有被牺牲」。</p>"
  )
  P.append(curves_html)

  # ---- §6 参数 ----
  P.append("<h2>6. 相关参数</h2>")
  cp = params["command"]
  P.append("<h3>6.1 命令空间</h3>")
  P.append(table(
    ["参数", "值"],
    [
      ["lin_vel_x", esc(cp["lin_vel_x"])],
      ["lin_vel_y", esc(cp["lin_vel_y"])],
      ["ang_vel_z", esc(cp["ang_vel_z"])],
      ["heading", esc(cp["heading"])],
      ["heading_command", esc(cp["heading_command"])],
      ["heading_control_stiffness", fmt(cp["heading_control_stiffness"], 2)],
      ["resampling_time_range (s)", esc(cp["resampling_time_range"])],
      ["rel_standing_envs", fmt(cp["rel_standing_envs"], 3)],
      ["rel_heading_envs", fmt(cp["rel_heading_envs"], 3)],
      ["rel_world_envs", fmt(cp["rel_world_envs"], 3)],
      ["rel_forward_envs", fmt(cp["rel_forward_envs"], 3)],
      ["rel_backward_envs", fmt(cp["rel_backward_envs"], 3)],
      ["rel_lateral_envs", fmt(cp["rel_lateral_envs"], 3)],
    ],
    cls=["", ""],
  ))
  P.append("<h3>6.2 奖励项</h3>")
  P.append(table(
    ["项", "函数", "权重", "参数"],
    [
      [esc(r["name"]), f"<code>{esc(r['func'])}</code>",
       f"<strong>{fmt(r['weight'], 4)}</strong>", f"<code>{esc(r['params'])}</code>"]
      for r in params["rewards"]
    ],
  ))
  P.append("<h3>6.3 课程项</h3>")
  if params["curriculum"]:
    P.append(table(
      ["课程", "阶段 (step, 参数)"],
      [[esc(k), f"<code>{esc(v)}</code>"] for k, v in params["curriculum"].items()],
    ))
  else:
    P.append("<p class='sub'>无课程项。</p>")
  P.append("<h3>6.4 仿真与回合</h3>")
  P.append(table(
    ["参数", "值"],
    [
      ["task", f"<code>{esc(params['task'])}</code>"],
      ["num_envs", esc(params["num_envs"])],
      ["episode_length_s", fmt(params["episode_length_s"], 1)],
      ["decimation", esc(params["decimation"])],
      ["sim timestep (s)", esc(params["sim_dt"])],
      ["控制周期 step_dt (s)", fmt(params["step_dt"], 4)],
      ["metrics 项数", esc(params["metrics_count"])],
      ["受控关节", f"<code>{esc(', '.join(params['joints']))}</code>"],
    ],
    cls=["", ""],
  ))

  # ---- §7 分析 ----
  if notes_html:
    P.append("<h2>7. 分析</h2>")
    P.append(notes_html)

  # ---- §8 CSV 说明 ----
  P.append("<h2>8. 数据文件与复现</h2>")
  P.append(
    "<p>每个档位一个 CSV，列契约与 "
    "<code>scripts/plot_logs/metrics_csv_dashboard.html</code> 一致：</p>"
  )
  P.append(
    "<pre>sample, sim_step, time_s, env_idx, network, "
    "&lt;指标列…&gt;</pre>"
  )
  P.append(
    "<ul>"
    "<li><code>time_s</code> 必须存在且<strong>单调递增</strong>（dashboard 会校验）；</li>"
    "<li>指标列按前缀被 dashboard 归类，本报告录了其中 8 族："
    "<code>joint_position_rad/</code>、"
    "<code>joint_target_rad/</code>、<code>joint_velocity_rad_s/</code>、"
    "<code>joint_torque_nm/</code>、<code>joint_torque_ratio/</code>、"
    "<code>joint_velocity_ratio/</code>、<code>joint_accel_rad_s2/</code>、"
    "<code>joint_power_w/</code>；</li>"
    "<li>dashboard 还认第 9 族 <code>joint_inertia_torque_nm/</code>，"
    "<strong>本报告故意没录</strong>：它要每个控制步为每个 env 算一次完整质量矩阵"
    "（<code>mj_fullM</code>），2048 env × 25000 迭代下开销实打实，"
    "而它承载的信息基本能从扭矩与加速度两条曲线推出来。"
    "这是<strong>已知的缺口</strong>，不是遗漏。</li>"
    "<li><code>vel/</code> 开头的列是<strong>本报告专用的速度追踪主线</strong>"
    "（<code>vel/cmd_*</code> 命令、<code>vel/act_*</code> 实测、"
    "<code>vel/err_*</code> 带符号误差、<code>vel/pos_*_w</code> 世界坐标、"
    "<code>vel/yaw_w</code> 世界偏航、<code>vel/standing</code> 门控快照）。"
    "dashboard 不认这个前缀，会原样忽略，所以两种用途互不干扰。</li>"
    "</ul>"
  )
  P.append("<h3>复现命令</h3>")
  P.append(
    "<pre>"
    "# 1) 录 CSV（⚠️ 训练期间不要跑：第二个仿真进程会把内存换页拖垮两边）\n"
    "MUJOCO_GL=egl uv run python scripts/record_v03_metrics.py \\\n"
    f"    --checkpoint {esc(checkpoint)} \\\n"
    f"    --out docs/pikachu_v03/metrics --tag {esc(tag)} --seconds 8\n\n"
    "# 2) 出报告（HTML + PDF）\n"
    "uv run python scripts/v03_velocity_report.py \\\n"
    f"    --metrics-dir docs/pikachu_v03/metrics --tag {esc(tag)} \\\n"
    f"    --checkpoint {esc(checkpoint)} \\\n"
    "    --out docs/pikachu_v03/reports/velocity-report.html\n\n"
    "# 3) 看图（把 CSV 拖进浏览器里的 dashboard）\n"
    "xdg-open scripts/plot_logs/metrics_csv_dashboard.html"
    "</pre>"
  )

  P.append("</body></html>")
  return "".join(P)


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------


def export_pdf(html_path: Path, pdf_path: Path) -> bool:
  """headless Chrome 导出 PDF。

  本机没有 weasyprint / wkhtmltopdf / pandoc；Chrome 是唯一装好的通路。
  ``--no-pdf-header-footer`` 去掉默认的页眉页脚（会印出 file:// 路径和日期）。
  """
  pdf_path.parent.mkdir(parents=True, exist_ok=True)
  cmd = [
    "google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
    "--no-pdf-header-footer", "--virtual-time-budget=8000",
    f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri(),
  ]
  try:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
  except FileNotFoundError:
    print("[report] 找不到 google-chrome，跳过 PDF", file=sys.stderr)
    return False
  if proc.returncode != 0 or not pdf_path.exists():
    print(f"[report] PDF 导出失败 rc={proc.returncode}\n{proc.stderr[-800:]}", file=sys.stderr)
    return False
  return True


# --------------------------------------------------------------------------


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--metrics-dir", default="docs/pikachu_v03/metrics")
  ap.add_argument("--tag", required=True, help="CSV 文件名里的 tag")
  ap.add_argument("--checkpoint", default="", help="写进报告抬头")
  ap.add_argument("--task", default="Mjlab-Velocity-Flat-Pikachu-V03")
  ap.add_argument("--out", default="docs/pikachu_v03/reports/velocity-report.html")
  ap.add_argument("--log-root", default="logs/rsl_rl/pikachu_v03_velocity")
  ap.add_argument("--curve-cache", default="docs/pikachu_v03/reports/.curve-cache.json")
  ap.add_argument("--notes", default="", help="Markdown 分析正文，插到 §7")
  ap.add_argument("--no-curves", action="store_true", help="跳过 tensorboard（慢）")
  ap.add_argument("--no-pdf", action="store_true")
  args = ap.parse_args()

  from datetime import datetime

  metrics_dir = Path(args.metrics_dir)
  clips: dict[str, dict[str, np.ndarray]] = {}
  for clip in CLIP_ORDER:
    p = metrics_dir / f"metrics_v03_{clip}_{args.tag}.csv"
    if p.exists():
      clips[clip] = load_clip(p)
      print(f"[report] {clip}: {p.name} ({len(clips[clip]['time_s'])} 行)")
    else:
      print(f"[report] {clip}: 缺 {p.name}，跳过", file=sys.stderr)
  if not clips:
    print(f"[report] {metrics_dir} 里没有 tag={args.tag} 的 CSV", file=sys.stderr)
    return 1

  per_clip = [analyze_clip(c, clips[c]) for c in CLIP_ORDER if c in clips]
  # 录制器旁挂的索引：补上 CSV 里没有的整档信息（终止步、步数、检查点）。
  index_path = metrics_dir / f"metrics_v03_{args.tag}_index.json"
  if index_path.exists():
    idx = json.loads(index_path.read_text(encoding="utf-8"))
    by_clip = {c["clip"]: c for c in idx.get("clips", [])}
    for rec in per_clip:
      meta = by_clip.get(rec["clip"], {})
      rec["terminated_at"] = meta.get("terminated_at")
      rec["steps"] = meta.get("steps")
      rec["requested_seconds"] = idx.get("seconds_per_clip")
    if idx.get("checkpoint") and not args.checkpoint:
      args.checkpoint = idx["checkpoint"]
    print(f"[report] 索引 {index_path.name}: "
          f"{sum(1 for c in by_clip.values() if c.get('terminated_at'))} 档被终止")
  else:
    print(f"[report] 无索引 {index_path.name}（终止信息缺失，其余不受影响）", file=sys.stderr)

  jrows = analyze_joints(clips)
  params = collect_params(args.task)

  charts: dict[str, str] = {}
  for clip in clips:
    uri = chart_tracking(clip, clips[clip])
    if uri:
      charts[f"track_{clip}"] = uri
  uri = chart_separation(clips)
  if uri:
    charts["separation"] = uri
  charts["joints"] = chart_joints(jrows)

  curves_html = ""
  if not args.no_curves:
    try:
      curves = read_curves(Path(args.log_root), Path(args.curve_cache))
      uri = chart_curves(curves)
      if uri:
        curves_html = (f"<figure><img src='{uri}' alt='训练曲线'>"
                       f"<figcaption>r04 / r05 / r06 关键标量对比</figcaption></figure>")
    except Exception as exc:  # tensorboard 读失败不该让整份报告出不来
      print(f"[report] 训练曲线跳过：{exc}", file=sys.stderr)

  notes_html = ""
  if args.notes:
    import markdown

    notes_html = markdown.markdown(
      Path(args.notes).read_text(encoding="utf-8"), extensions=["tables", "fenced_code"]
    )

  html = build_html(
    tag=args.tag,
    checkpoint=args.checkpoint or "(未指定)",
    clips=clips,
    per_clip=per_clip,
    jrows=jrows,
    params=params,
    curves_html=curves_html,
    notes_html=notes_html,
    charts=charts,
    generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
  )

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(html, encoding="utf-8")
  print(f"[report] HTML -> {out} ({out.stat().st_size // 1024} KiB)")

  if not args.no_pdf:
    pdf = out.with_suffix(".pdf")
    if export_pdf(out, pdf):
      print(f"[report] PDF  -> {pdf} ({pdf.stat().st_size // 1024} KiB)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
