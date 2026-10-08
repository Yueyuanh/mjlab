"""把 Pikachu v03 策略的**逐关节遥测**录成 CSV（无头、无 viewer）。

产出直接被 ``scripts/plot_logs/metrics_csv_dashboard.html`` 消费（列契约见下），
同时带一条 dashboard **没有**的主线：速度追踪（``vel/`` 前缀，见
``mjlab/tasks/velocity/mdp/metrics.py``）。

## 列契约（dashboard 的硬要求）

    sample, sim_step, time_s, env_idx, network, <指标列...>

  - ``time_s`` **必须存在且单调递增**（dashboard 会抛错校验），值 = 控制周期 × 步数；
  - ``network`` 是数据集显示名，这里放档位 tag；
  - 指标列名以 ``joint_position_rad/`` / ``joint_target_rad/`` /
    ``joint_velocity_rad_s/`` / ``joint_torque_nm/`` / ``joint_torque_ratio/`` /
    ``joint_velocity_ratio/`` 等前缀开头，dashboard 按前缀归类；
  - 左/右识别靠 ``left_``/``right_`` 前缀，所以 v03 的关节名（``left_hip_pitch_joint``）
    天然满足，不需要改名。

## 命令怎么固定

**在构造 env 之前**把 ``twist`` 的 ranges 收成一个点，并把所有 ``rel_*_envs``
归零 —— 与 ``record_v03_video.py`` / ``eval_v03_velocity.py`` **同一套**做法，
所以三者的档位数字可以直接互相印证。⚠️ 必须 ``curriculum.pop("command_vel")``，
否则课程项每步把 ranges 覆写回默认值，六个档会录出六段一模一样的片子。

观测噪声关掉（``enable_corruption=False``）：报告要报的是**部署时会看到**的
追踪精度，不是训练时的带噪条件。

用法::

    cd mjlab
    # ⚠️ 训练期间不要跑（第二个仿真进程会把内存换页拖垮两边，见 README 坑 3）
    MUJOCO_GL=egl uv run python scripts/record_v03_metrics.py \
        --checkpoint logs/rsl_rl/pikachu_v03_velocity/<run>/model_29999.pt \
        --out docs/pikachu_v03/metrics --tag r06-best \
        --clips stand,fwd,fwd_fast,back,lat,turn --seconds 8
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

# 档位定义与录像脚本**共用同一份**（含 fwd_fast）。scripts/ 在 sys.path[0]，
# 直接 import 保证「固定命令」的语义不会两边漂移。
from record_v03_video import CLIPS

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

# 与 Viser MetricsCsvRecorder / mjlab.scripts.record_metrics_cli 逐字一致的前 5 列。
_BASE_COLUMNS = ("sample", "sim_step", "time_s", "env_idx", "network")


def build_policy(task: str, checkpoint: str, device: str, env_cfg):
  """建 raw env + 推理策略（raw env 才有 metrics_manager 和 5 元组 step）。"""
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  # runner 要的是 dict 不是 dataclass（与 record_v03_video.py 一致）。
  runner = runner_cls(wrapped, asdict(load_rl_cfg(task)), log_dir=None, device=device)
  # 只载 actor：评测不需要 critic/optimizer，省内存（本机 swap 已满）。
  runner.load(
    checkpoint,
    load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False},
  )
  runner.alg.eval_mode()
  return env, runner.get_inference_policy(device=device)


def make_env_cfg(task: str, clip: str, seed: int):
  """单 env + 固定命令 + 无噪声 + 无课程，与录像/评测脚本同一套。"""
  vx, vy, wz = CLIPS[clip]
  cfg = load_env_cfg(task, play=False)
  cfg.scene.num_envs = 1
  cfg.episode_length_s = 1.0e9  # 不因超时截断，只可能被「摔倒」终止
  cfg.observations["actor"].enable_corruption = False
  cfg.events.pop("push_robot", None)
  # ⚠️ 必须弹掉，否则 ranges 被课程项冲回默认值（六个档会一模一样）。
  cfg.curriculum.pop("command_vel", None)
  cfg.seed = seed  # base_com 是 startup 事件，构造时就抽完了

  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = (vx, vx)
  twist.ranges.lin_vel_y = (vy, vy)
  twist.ranges.ang_vel_z = (wz, wz)
  twist.heading_command = False
  twist.ranges.heading = None  # heading_command=False 时非 None 会直接抛错
  for attr in (
    "rel_standing_envs",
    "rel_heading_envs",
    "rel_world_envs",
    "rel_forward_envs",
    "rel_backward_envs",
    "rel_lateral_envs",
  ):
    setattr(twist, attr, 0.0)
  return cfg


def record_clip(
  task: str,
  checkpoint: str,
  clip: str,
  steps: int,
  device: str,
  seed: int,
  tag: str,
  out_path: Path,
) -> dict:
  """跑一个档位并写 CSV，返回该档的摘要（供终端打印）。"""
  env, policy = build_policy(task, checkpoint, device, make_env_cfg(task, clip, seed))
  try:
    obs, _ = env.reset()
    step_dt = float(env.step_dt)
    metric_names = list(env.metrics_manager.active_terms)
    n = 0
    terminated_at: int | None = None

    with out_path.open("w", encoding="utf-8", newline="") as stream:
      writer = csv.writer(stream)
      writer.writerow([*_BASE_COLUMNS, *metric_names])
      for step in range(1, steps + 1):
        with torch.no_grad():
          action = policy(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        terms = dict(env.metrics_manager.get_active_iterable_terms(0))
        row = [
          step - 1,
          step,
          (step - 1) * step_dt,
          0,
          tag,
          *(terms.get(name, [float("nan")])[0] for name in metric_names),
        ]
        writer.writerow([f"{v:.9g}" if isinstance(v, float) else v for v in row])
        n = step
        if bool(terminated[0]) or bool(truncated[0]):
          terminated_at = step
          break
    return {
      "clip": clip,
      "path": out_path,
      "steps": n,
      "seconds": n * step_dt,
      "terminated_at": terminated_at,
      "metric_names": metric_names,
    }
  finally:
    env.close()


# dashboard 契约里、本任务应当录到的 8 族关节列。缺任何一族都说明
# cfg.metrics 没挂上（不是「这族本来就没有」），要立刻报出来。
_EXPECTED_FAMILIES = (
  "joint_position_rad/",
  "joint_target_rad/",
  "joint_velocity_rad_s/",
  "joint_torque_nm/",
  "joint_torque_ratio/",
  "joint_velocity_ratio/",
  "joint_accel_rad_s2/",
  "joint_power_w/",
)


def check_columns(rec: dict) -> str | None:
  """扫一遍刚写出的 CSV，把**整列全空/全 NaN** 与**缺族**的指标报出来。

  为什么值得单独一步：一个写错的指标函数（取错字段、形状不对）不会让录制
  崩掉，只会安静地写一整列 ``nan``；而一个没挂上的指标干脆连列都不出现。
  两种都不会让脚本报错，却都会让这份 CSV 被当成「有效数据」一路送进报告。
  这里在**写完之后立刻**自检，把问题暴露在终端里。

  返回一句警告（没问题则 None）。
  """
  with rec["path"].open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
  if not rows:
    return f"{rec['clip']}: CSV 没有数据行"
  bad: list[str] = []
  for name in rec["metric_names"]:
    vals = [r.get(name) for r in rows]
    if all(v in (None, "", "nan") for v in vals):
      bad.append(name)
  missing = [
    p for p in _EXPECTED_FAMILIES if not any(n.startswith(p) for n in rec["metric_names"])
  ]
  msgs = []
  if bad:
    head = ", ".join(bad[:6]) + (" …" if len(bad) > 6 else "")
    msgs.append(f"{len(bad)} 列整列无值 —— {head}")
  if missing:
    msgs.append(f"缺 {len(missing)} 族 —— {', '.join(missing)}")
  return f"{rec['clip']}: " + "；".join(msgs) if msgs else None


def summarize(rec: dict) -> str:
  """从刚写出的 CSV 里算该档的追踪摘要（不看奖励，只看物理量）。"""
  with rec["path"].open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
  if not rows:
    return f"{rec['clip']}: 无数据"

  def col(name: str) -> list[float]:
    return [float(r[name]) for r in rows if r.get(name) not in (None, "")]

  def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")

  # 丢掉前 0.5 s 的起步瞬态再统计。
  skip = min(len(rows) // 8, 25)
  body = rows[skip:]

  def colb(name: str) -> list[float]:
    return [float(r[name]) for r in body if r.get(name) not in (None, "")]

  cmd_x, act_x = mean(colb("vel/cmd_x")), mean(colb("vel/act_x"))
  cmd_y, act_y = mean(colb("vel/cmd_y")), mean(colb("vel/act_y"))
  cmd_w, act_w = mean(colb("vel/cmd_yaw")), mean(colb("vel/act_yaw"))
  err_n = mean(colb("vel/err_xy_norm"))
  px, py = col("vel/pos_x_w"), col("vel/pos_y_w")
  net = ((px[-1] - px[0]) ** 2 + (py[-1] - py[0]) ** 2) ** 0.5 if px and py else float("nan")
  return (
    f"{rec['clip']:9s} {rec['seconds']:4.1f}s "
    f"cmd=({cmd_x:+.2f},{cmd_y:+.2f},{cmd_w:+.2f}) "
    f"act=({act_x:+.2f},{act_y:+.2f},{act_w:+.2f}) "
    f"err_xy={err_n:.3f} 净位移={net:.2f}m"
    + (f"  ⚠️ 第 {rec['terminated_at']} 步终止" if rec["terminated_at"] else "")
  )


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="Mjlab-Velocity-Flat-Pikachu-V03")
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--out", required=True, help="CSV 输出目录")
  ap.add_argument("--tag", default="", help="network 列 / 文件名里的标识")
  ap.add_argument("--clips", default="stand,fwd,fwd_fast,back,lat,turn")
  ap.add_argument("--seconds", type=float, default=8.0, help="每档录多少秒")
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--seed", type=int, default=42)
  args = ap.parse_args()

  want = [c.strip() for c in args.clips.split(",") if c.strip()]
  bad = [c for c in want if c not in CLIPS]
  if bad:
    raise SystemExit(f"未知档位 {bad}，可选 {list(CLIPS)}")

  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)
  tag = args.tag or Path(args.checkpoint).parent.name

  print(f"[record_v03_metrics] ckpt={args.checkpoint}")
  print(f"[record_v03_metrics] tag={tag}  clips={want}  {args.seconds}s/档")
  results = []
  for clip in want:
    t0 = time.time()
    # step_dt = decimation × timestep；先建一次 cfg 拿它，再换算步数。
    probe = make_env_cfg(args.task, clip, args.seed)
    step_dt = probe.decimation * probe.sim.mujoco.timestep
    steps = int(args.seconds / step_dt)
    dest = out_dir / f"metrics_v03_{clip}_{tag}.csv"
    rec = record_clip(
      args.task, args.checkpoint, clip, steps, args.device, args.seed, tag, dest
    )
    results.append(rec)
    print(f"[record_v03_metrics] {summarize(rec)}  "
          f"({rec['path'].name}, {rec['path'].stat().st_size // 1024} KiB, "
          f"{time.time() - t0:.1f}s)", flush=True)
    warn = check_columns(rec)
    if warn:
      print(f"[record_v03_metrics] ⚠️ 空列告警 {warn}", flush=True)

  print(f"\n[record_v03_metrics] 完成 {len(results)} 档 -> {out_dir}")

  # 旁挂一份索引：报告脚本要拿「哪档被终止了」「检查点是谁」这类 CSV 里
  # 没有的信息。不写进 CSV 是因为列契约要留给 dashboard（它按前缀分类，
  # 多一列非指标列没有坏处但也没有好处），而且这几项是**整档常量**，
  # 逐行重复 400 遍是噪声。
  index = {
    "checkpoint": str(args.checkpoint),
    "tag": tag,
    "seconds_per_clip": args.seconds,
    "seed": args.seed,
    "clips": [
      {
        "clip": r["clip"],
        "file": r["path"].name,
        "steps": r["steps"],
        "seconds": round(r["seconds"], 4),
        "terminated_at": r["terminated_at"],
        "n_metrics": len(r["metric_names"]),
      }
      for r in results
    ],
  }
  index_path = out_dir / f"metrics_v03_{tag}_index.json"
  index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
  print(f"  索引 -> {index_path.name}")

  print("  把 CSV 拖进 scripts/plot_logs/metrics_csv_dashboard.html 即可看图；")
  print("  vel/ 开头的列 dashboard 会忽略（它只认关节/基座族），由分析脚本读取。")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
