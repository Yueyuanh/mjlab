"""对 V031 五档消融的 checkpoint 跑同口径评测，汇总成一张对比表。

每档用 ``load_rl_cfg(task).experiment_name`` 定位 ``logs/rsl_rl/<exp>/`` 下
**最新的时间戳子目录**，再对其中的 ``model_<ckpt>.pt`` 逐个调用
``scripts/eval_v031_symmetry.py``（评测协议、指标口径全部复用同一份代码，
不在这里重写，避免两套口径漂移）。

**不要在训练还在跑的时候执行**：评测与训练抢同一块 GPU，8 GiB 上大概率 OOM，
且会拖慢训练。脚本启动时会检查有无存活的 train 进程并警告。

用法（在 mjlab/ 下）：

    MUJOCO_GL=egl uv run python scripts/eval_v031_all_arms.py \\
        --checkpoints 1000,3000,4999 --seconds 8 --seeds 42,7 \\
        --out /tmp/v031_eval

产物：``<out>/<arm>/model_<ckpt>/eval.json`` 以及汇总表 ``<out>/summary.json``。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from mjlab.tasks.registry import load_rl_cfg

# 档名 -> 任务 ID。顺序即报告里的呈现顺序。
ARMS: tuple[tuple[str, str], ...] = (
  ("T6", "Mjlab-Velocity-Flat-Pikachu-V031-T6"),
  ("Clip", "Mjlab-Velocity-Flat-Pikachu-V031-Clip"),
  ("Sym", "Mjlab-Velocity-Flat-Pikachu-V031-Sym"),
  ("SymClip", "Mjlab-Velocity-Flat-Pikachu-V031-SymClip"),
  ("YawHuber", "Mjlab-Velocity-Flat-Pikachu-V031-YawHuber"),
)


def latest_run(exp_name: str, log_root: Path) -> Path | None:
  """``<log_root>/<exp_name>/`` 下最新的时间戳子目录。"""
  base = log_root / exp_name
  if not base.is_dir():
    return None
  runs = sorted(p for p in base.iterdir() if p.is_dir())
  return runs[-1] if runs else None


def warn_if_training() -> None:
  out = subprocess.run(
    ["pgrep", "-af", "uv run train"], capture_output=True, text=True
  ).stdout.strip()
  if out:
    print("!! 检测到训练进程仍在运行，评测会与它抢 GPU：", file=sys.stderr)
    for line in out.splitlines():
      print(f"   {line}", file=sys.stderr)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoints", default="1000,3000,4999")
  ap.add_argument("--seconds", type=float, default=8.0)
  ap.add_argument("--seeds", default="42,7")
  ap.add_argument("--out", default="/tmp/v031_eval")
  ap.add_argument("--log-root", default="logs/rsl_rl")
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument(
    "--arms",
    default="",
    help="逗号分隔的档名子集，空=全部（T6,Clip,Sym,SymClip,YawHuber）",
  )
  args = ap.parse_args()

  warn_if_training()

  ckpts = [int(c) for c in args.checkpoints.split(",") if c.strip()]
  wanted = {a for a in args.arms.split(",") if a.strip()}
  arms = [(n, t) for n, t in ARMS if not wanted or n in wanted]
  out_root = Path(args.out)
  log_root = Path(args.log_root)

  summary: dict[str, dict] = {}
  for arm, task in arms:
    run = latest_run(load_rl_cfg(task).experiment_name, log_root)
    if run is None:
      print(f"[{arm}] 找不到 run 目录，跳过", file=sys.stderr)
      continue
    print(f"\n===== {arm}  ({task})\n      run = {run}")
    for ck in ckpts:
      ckpt = run / f"model_{ck}.pt"
      if not ckpt.is_file():
        print(f"  model_{ck}.pt 不存在，跳过")
        continue
      outdir = out_root / arm / f"model_{ck}"
      cmd = [
        sys.executable,
        "scripts/eval_v031_symmetry.py",
        "--task",
        task,
        "--checkpoint",
        str(ckpt),
        "--out",
        str(outdir),
        "--device",
        args.device,
        "--seconds",
        str(args.seconds),
        "--seeds",
        args.seeds,
      ]
      r = subprocess.run(cmd)
      if r.returncode != 0:
        print(f"  评测失败 rc={r.returncode}", file=sys.stderr)
        continue
      summary.setdefault(arm, {})[f"model_{ck}"] = json.loads(
        (outdir / "eval.json").read_text()
      )

  out_root.mkdir(parents=True, exist_ok=True)
  (out_root / "summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False)
  )

  print("\n\n================ 汇总 ================")
  print(
    f"{'档':<10}{'ckpt':>7}{'摔倒':>6}{'°/m':>8}{'yawRMS':>8}"
    f"{'|hr_d|':>9}{'越界率':>8}{'镜像等变':>9}"
  )
  print("-" * 66)
  for arm, _ in arms:
    for key, e in summary.get(arm, {}).items():
      a = e["agg_straight"]
      me = e.get("mirror_equiv", {}).get("ratio", float("nan"))
      print(
        f"{arm:<10}{key.replace('model_', ''):>7}"
        f"{str(a['fell_any']):>6}"
        f"{a['heading_drift_deg_per_m_mean']:>8.1f}"
        f"{a['yaw_rate_rms_mean']:>8.3f}"
        f"{a['hr_d_absmax']:>9.4f}"
        f"{a['target_violation_max']:>8.3f}"
        f"{me:>9.3f}"
      )
  print(f"\n汇总 JSON: {out_root / 'summary.json'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
