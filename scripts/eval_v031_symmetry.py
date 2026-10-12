"""V031 对称性 / 偏航的**固定评测协议**（可跨 checkpoint 对比）。

诊断脚本 ``diag_v031_rollout.py`` 只跑 fwd/stand 两条命令、且只报原始时间序列；
本脚本把评测口径固定下来，让"基线 / 裁剪 / 镜像 / 两者"四个档能在同一把尺子
下比较。核心指标都围绕**镜像对称性**定义：

- ``hr_d``：左右 ``hip_roll`` 的同向分量 ``(mean qL + mean qR) / 2``。
  **镜像对称的步态必然满足 qL = -qR，所以 d 的理论值就是 0** —— 它是"策略
  是否学出单侧解"最直接的标量。站立时的 d 也应为 0。
- ``hr_p2p_ratio``：``std(qL) / std(qR)``，对称步态应为 1。
- ``heading_drift_deg_per_m``：``|Δyaw| / 路程``。直行命令下这是"走弧线"的
  直接度量，比单纯看 yaw 漂移更能反映"每走一米偏多少"。
- ``lateral_ratio``：``|Δy| / |Δx|``，直行命令下的侧向漂移占比。
- ``yaw_rate_rms``：机体 z 角速度 RMS，零偏航命令下应为 0。
- ``target_violation``：处理后关节目标越过硬限位的采样比例（安全边界是否生效）。
- ``mirror_equiv``：``policy(mirror(obs))`` 与 ``mirror(policy(obs))`` 的逐元素
  平均绝对差 / 策略输出尺度。0 = 完全镜像等变。

用法::

    MUJOCO_GL=egl uv run python scripts/eval_v031_symmetry.py \
        --task Mjlab-Velocity-Flat-Pikachu-V031-Sym \
        --checkpoint logs/rsl_rl/.../model_4999.pt \
        --out /tmp/v031_eval/sym
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.config.pikachu_v031.symmetry import (
  JOINT_NAMES,
  mirror_augmentation,
)

# 评测命令表：(名字, (vx, vy, wz))。
COMMANDS: tuple[tuple[str, tuple[float, float, float]], ...] = (
  ("stand", (0.0, 0.0, 0.0)),
  ("fwd", (0.4, 0.0, 0.0)),
  ("back", (-0.3, 0.0, 0.0)),
  ("left", (0.0, 0.3, 0.0)),
  ("right", (0.0, -0.3, 0.0)),
  ("turnL", (0.0, 0.0, 0.5)),
  ("turnR", (0.0, 0.0, -0.5)),
)

# hip_roll 的硬限位（见 pikachu_v031_constants / env_cfgs 注释）。
HIP_ROLL_LIMIT = 0.26


def quat_to_rpy(q: np.ndarray) -> np.ndarray:
  """wxyz 四元数 -> (roll, pitch, yaw)。``root_link_quat_w`` 是 wxyz 序。"""
  w, x, y, z = q
  roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
  yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  return np.array([roll, pitch, yaw])


def _actor(obs):
  """取 actor 观测组。wrapper 返回 ``TensorDict``，裸 env 返回 dict。"""
  if isinstance(obs, (dict, TensorDict)):
    return obs["actor"]
  return obs


def build(
  task: str,
  checkpoint: str,
  device: str,
  num_envs: int,
  seed: int,
  clip_override: float | None = None,
):
  cfg = load_env_cfg(task, play=False)
  cfg.scene.num_envs = num_envs
  cfg.episode_length_s = 1.0e9
  cfg.observations["actor"].enable_corruption = False
  cfg.events.pop("push_robot", None)
  cfg.curriculum.pop("command_vel", None)
  cfg.seed = seed
  # 固定命令由 set_command 逐条覆盖，这里先把随机化关掉。
  twist = cfg.commands["twist"]
  twist.heading_command = False
  twist.ranges.heading = None
  twist.resampling_time_range = (1.0e9, 1.0e9)
  for attr in (
    "rel_standing_envs",
    "rel_heading_envs",
    "rel_world_envs",
    "rel_forward_envs",
    "rel_backward_envs",
    "rel_lateral_envs",
  ):
    setattr(twist, attr, 0.0)
  cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)

  agent_cfg = load_rl_cfg(task)
  # **必须与训练时一致**：``train.py`` 把 ``cfg.agent.clip_actions`` 传给
  # ``RslRlVecEnvWrapper``。不传的话，Clip / SymClip 档会在**没有裁剪**的动作
  # 管线下被评测 —— 量到的不是它训练出来的那个策略。
  clip_actions = agent_cfg.clip_actions if clip_override is None else clip_override

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=clip_actions)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), log_dir=None, device=device)
  runner.load(
    checkpoint,
    load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False},
  )
  runner.alg.eval_mode()
  # 返回 **wrapper** 而不是裸 env：``rollout`` 必须走 ``wrapped.step()``，否则
  # ``clip_actions`` 完全不生效（裸 env 的 step 不做裁剪）。需要看内部状态时用
  # ``env.unwrapped``。
  return wrapped, runner.get_inference_policy(device=device)


def _raw(env):
  """取裸 ``ManagerBasedRlEnv``（wrapper 没有属性转发）。"""
  return getattr(env, "unwrapped", env)


def set_command(env, cmd: tuple[float, float, float]) -> None:
  term = _raw(env).command_manager.get_term("twist")
  term.cfg.ranges.lin_vel_x = (cmd[0], cmd[0])
  term.cfg.ranges.lin_vel_y = (cmd[1], cmd[1])
  term.cfg.ranges.ang_vel_z = (cmd[2], cmd[2])


def rollout(env, policy, seconds: float, seed: int) -> dict:
  env.seed(seed)
  obs, _ = env.reset()
  robot = _raw(env).scene["robot"]
  term = _raw(env).action_manager.get_term("joint_pos")
  step_dt = float(_raw(env).step_dt)
  steps = int(round(seconds / step_dt))
  rec: dict[str, list] = {
    k: []
    for k in (
      "pos_w",
      "rpy",
      "lin_vel_b",
      "ang_vel_b",
      "q",
      "action",
      "target",
    )
  }
  done_at = None
  for i in range(steps):
    with torch.no_grad():
      action = policy(obs)
    # ``wrapped.step`` 返回 4 元组，``dones`` 已合并 terminated|truncated，
    # 且裁剪就发生在这里面。
    obs, _, dones, _ = env.step(action)
    d = robot.data
    rec["pos_w"].append(d.root_link_pos_w[0].cpu().numpy().copy())
    rec["rpy"].append(quat_to_rpy(d.root_link_quat_w[0].cpu().numpy()))
    rec["lin_vel_b"].append(d.root_link_lin_vel_b[0].cpu().numpy().copy())
    rec["ang_vel_b"].append(d.root_link_ang_vel_b[0].cpu().numpy().copy())
    rec["q"].append(d.joint_pos[0].cpu().numpy().copy())
    # 记**裁剪前**的原始策略输出：``action_clip_frac`` 要量的是"裁剪会不会
    # 咬到"；裁剪后的效果已经体现在 ``target`` 里了。
    rec["action"].append(action[0].detach().cpu().numpy().copy())
    processed = getattr(term, "_processed_actions", None)
    if processed is not None:
      rec["target"].append(processed[0].detach().cpu().numpy().copy())
    if bool(dones[0]):
      done_at = i
      break
  out = {k: np.asarray(v) for k, v in rec.items()}
  out["done_at"] = done_at
  out["step_dt"] = step_dt
  # ``_processed_actions`` 的列序由 action term 自己决定，是**第三种**顺序：
  #   Entity.joint_names   = right 7 个, left 7 个（含被动的 knee / knee_mid）
  #   Entity.actuator_names= right 髋×4, left 髋×4, right_ankle, left_ankle
  #   term.target_names    = right_hip_pitch/roll/yaw/ankle/knee_up, left 同序
  # 三者互不相同，所以只能从 term 读，不能硬编码下标。
  out["target_names"] = list(term.target_names)
  return out


def summarize(r: dict) -> dict:
  rpy, q, av = r["rpy"], r["q"], r["ang_vel_b"]
  pos = r["pos_w"]
  n = len(rpy)
  iL, iR = (
    JOINT_NAMES.index("left_hip_roll_joint"),
    JOINT_NAMES.index("right_hip_roll_joint"),
  )
  qL, qR = q[:, iL], q[:, iR]
  path = (
    float(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1).sum()) if n > 1 else 0.0
  )
  dyaw = float(abs(rpy[-1, 2] - rpy[0, 2])) if n else 0.0
  dx = float(pos[-1, 0] - pos[0, 0]) if n else 0.0
  dy = float(pos[-1, 1] - pos[0, 1]) if n else 0.0

  tgt = r["target"]
  violation = float("nan")
  if tgt.size:
    # **顺序陷阱**：``_processed_actions`` 的列是 action term 自己的顺序
    # （``term.target_names``），与 ``Entity.joint_names``（14 项）和
    # ``Entity.actuator_names``（10 项）**都不一样**。用 JOINT_NAMES 的下标去
    # 索引它，左腿那两个下标会落到踝 / knee_up 上（scale 0.35 / 0.40 > 0.26），
    # 把别的关节的越界算成 hip_roll 的越界 —— 这个 bug 一度让越界率虚高到 0.85。
    target_names = r["target_names"]
    if tgt.shape[-1] != len(target_names):
      raise ValueError(f"处理后目标维度应为 {len(target_names)}，实际 {tgt.shape[-1]}")
    idx = [
      i
      for i, nm in enumerate(target_names)
      if nm.endswith(("hip_roll_joint", "hip_yaw_joint"))
    ]
    if not idx:
      raise ValueError(f"target_names 里找不到 hip_roll / hip_yaw：{target_names}")
    sel = tgt[:, idx]
    violation = float((np.abs(sel) > HIP_ROLL_LIMIT).any(axis=1).mean())

  return {
    "seconds": float(n * r["step_dt"]),
    "fell": r["done_at"] is not None,
    "roll_mean": float(rpy[:, 0].mean()),
    "pitch_mean": float(rpy[:, 1].mean()),
    "yaw_drift_deg": float(np.degrees(dyaw)),
    "yaw_rate_rms": float(np.sqrt(np.mean(av[:, 2] ** 2))),
    "yaw_rate_mean": float(av[:, 2].mean()),
    "path_len_m": path,
    "dx_m": dx,
    "dy_m": dy,
    "heading_drift_deg_per_m": float(np.degrees(dyaw) / path) if path > 0.05 else 0.0,
    "lateral_ratio": float(abs(dy) / abs(dx)) if abs(dx) > 0.05 else float("nan"),
    "lin_vel_err_xy": float(
      np.sqrt(np.mean((r["lin_vel_b"][:, 0] - np.mean(r["lin_vel_b"][:, 0])) ** 2))
    ),
    "hr_L_mean": float(qL.mean()),
    "hr_R_mean": float(qR.mean()),
    "hr_d": float((qL.mean() + qR.mean()) / 2.0),
    "hr_s": float((qL.mean() - qR.mean()) / 2.0),
    "hr_L_std": float(qL.std()),
    "hr_R_std": float(qR.std()),
    "hr_p2p_ratio": float(qL.std() / qR.std()) if qR.std() > 1e-6 else float("nan"),
    "hr_L_absmax": float(np.abs(qL).max()),
    "hr_R_absmax": float(np.abs(qR).max()),
    "target_violation": violation,
    "action_absmax": float(np.abs(r["action"]).max()),
    "action_clip_frac": float((np.abs(r["action"]) > 1.0).mean()),
  }


def mirror_equivariance(env, policy, n_samples: int = 200) -> dict:
  obs, _ = env.reset()
  collected = []
  for _ in range(n_samples):
    with torch.no_grad():
      action = policy(obs)
    collected.append(_actor(obs)[0].detach().cpu().numpy().copy())
    obs, _, dones, _ = env.step(action)
    if bool(dones[0]):
      break
  x = np.asarray(collected)
  if x.shape[0] == 0:
    return {"n_samples": 0}
  t = torch.tensor(x, dtype=torch.float32, device=_raw(env).device)
  b = t.shape[0]
  # 用与训练完全相同的镜像函数，保证口径一致。
  td = TensorDict({"actor": t}, batch_size=[b], device=_raw(env).device)
  td_m, _ = mirror_augmentation(env, td, None)
  assert td_m is not None
  with torch.no_grad():
    # [原观测的动作; 镜像观测的动作]
    a_all = policy({"actor": td_m["actor"]}).cpu().numpy()
  # 原动作的镜像
  _, a_orig_mir_t = mirror_augmentation(
    env, None, torch.tensor(a_all[:b], device=_raw(env).device)
  )
  assert a_orig_mir_t is not None
  a_orig_mir = a_orig_mir_t[b:].cpu().numpy()
  diff = a_all[b:] - a_orig_mir
  scale = float(a_all[:b].std())
  return {
    "n_samples": int(x.shape[0]),
    "mean_abs_diff": float(np.abs(diff).mean()),
    "max_abs_diff": float(np.abs(diff).max()),
    "act_std": scale,
    "ratio": float(np.abs(diff).mean() / max(scale, 1e-9)),
    "per_dim": np.abs(diff).mean(axis=0).tolist(),
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", required=True)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--seconds", type=float, default=8.0)
  ap.add_argument("--seeds", default="42,7")
  ap.add_argument("--commands", default="", help="逗号分隔的命令名，空=全部")
  ap.add_argument("--skip-mirror", action="store_true")
  ap.add_argument(
    "--clip",
    type=float,
    default=None,
    help=(
      "覆盖动作裁剪上限（默认用任务自己的 clip_actions）。"
      "用来单独验证「把裁剪加到未裁剪训练出来的策略上会怎样」："
      "传 1.0 即可在 T6 checkpoint 上强制开裁剪。"
    ),
  )
  args = ap.parse_args()

  outdir = Path(args.out)
  outdir.mkdir(parents=True, exist_ok=True)
  seeds = [int(s) for s in args.seeds.split(",")]
  wanted = set(args.commands.split(",")) if args.commands else None
  cmds = [c for c in COMMANDS if wanted is None or c[0] in wanted]

  env, policy = build(
    args.task, args.checkpoint, args.device, 1, seeds[0], clip_override=args.clip
  )
  rows: list[dict] = []
  try:
    print(
      f"{'命令':<8}{'seed':>5}{'摔倒':>6}{'Δyaw°':>9}{'°/m':>8}"
      f"{'yawRMS':>8}{'侧漂比':>8}{'hr_d':>9}{'p2p比':>7}{'越界率':>8}"
    )
    print("-" * 78)
    for name, cmd in cmds:
      set_command(env, cmd)
      for sd in seeds:
        s = summarize(rollout(env, policy, args.seconds, sd))
        s["command"] = name
        s["cmd"] = list(cmd)
        s["seed"] = sd
        rows.append(s)
        print(
          f"{name:<8}{sd:>5}{str(s['fell']):>6}{s['yaw_drift_deg']:>9.1f}"
          f"{s['heading_drift_deg_per_m']:>8.1f}{s['yaw_rate_rms']:>8.3f}"
          f"{s['lateral_ratio']:>8.2f}{s['hr_d']:>+9.4f}"
          f"{s['hr_p2p_ratio']:>7.3f}{s['target_violation']:>8.3f}"
        )
    me = {}
    if not args.skip_mirror:
      set_command(env, (0.4, 0.0, 0.0))
      me = mirror_equivariance(env, policy)
      print(
        f"\n镜像等变误差 / 输出尺度 = {me.get('ratio', float('nan')):.3f} "
        f"(样本 {me.get('n_samples', 0)})"
      )
  finally:
    env.close()

  # 跨命令聚合：只对"直行类"命令（stand/fwd/back）看对称性与偏航。
  straight = [r for r in rows if r["command"] in ("stand", "fwd", "back")]
  agg = {
    "task": args.task,
    "checkpoint": args.checkpoint,
    "clip_actions": args.clip
    if args.clip is not None
    else load_rl_cfg(args.task).clip_actions,
    "seconds": args.seconds,
    "seeds": seeds,
    "rows": rows,
    "mirror_equiv": me,
    "agg_straight": {
      "hr_d_mean": float(np.mean([r["hr_d"] for r in straight])),
      "hr_d_absmax": float(np.max(np.abs([r["hr_d"] for r in straight]))),
      "heading_drift_deg_per_m_mean": float(
        np.mean([r["heading_drift_deg_per_m"] for r in straight])
      ),
      "yaw_rate_rms_mean": float(np.mean([r["yaw_rate_rms"] for r in straight])),
      "target_violation_max": float(
        np.nanmax([r["target_violation"] for r in rows])
        if any(r["target_violation"] == r["target_violation"] for r in rows)
        else float("nan")
      ),
      "fell_any": bool(any(r["fell"] for r in rows)),
    },
  }
  (outdir / "eval.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False))
  a = agg["agg_straight"]
  print("\n=== 聚合（stand/fwd/back）===")
  print(
    f"  hip_roll 同向分量 |d| 均值 = {abs(a['hr_d_mean']):.4f} "
    f"(最大 {a['hr_d_absmax']:.4f})   [镜像对称应为 0]"
  )
  print(f"  每米偏航漂移 = {a['heading_drift_deg_per_m_mean']:.1f} °/m")
  print(f"  yaw_rate RMS = {a['yaw_rate_rms_mean']:.3f} rad/s")
  print(f"  目标越界率最大 = {a['target_violation_max']:.4f}")
  print(f"  任一条摔倒 = {a['fell_any']}")
  print(f"结果已写出 -> {outdir / 'eval.json'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
