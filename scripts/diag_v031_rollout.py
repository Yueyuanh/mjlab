"""V031 策略 rollout 深度诊断：base 姿态、左右 hip_roll 的 DC/AC 分解、镜像等变性。

现有的 ``record_v03_metrics.py`` 录的指标里**没有 base 的 roll/pitch**，而
"走路向一侧倾斜"恰恰要看这个。本脚本补上，并回答两个问题：

  1. **非对称是"静态偏置"还是"动态摆动"？** 把左右 hip_roll 分解成
     对称分量 s = (qL - qR)/2（双腿开合）与同向分量 d = (qL + qR)/2
     （双腿同向 = 重心横移）。走路时的 d 若长期非零，就是"向一侧偏"。
  2. **策略是不是镜像等变的？** 取一条真实观测轨迹，构造镜像观测
     obs' = mirror(obs)，比较 ``policy(obs')`` 与 ``mirror(policy(obs))``。
     完全等变的策略两者逐元素相等（差 0）；差值大 = 策略学到了非对称解。

镜像约定用 ``scripts/diag_v031_mirror_equiv.py`` 判定出的真值表：
hip_pitch/hip_roll/hip_yaw/knee/ankle 反号，knee_up/knee_mid 同号。

**顺序陷阱（本脚本曾踩）**：``Entity.joint_names`` 是 **right 在前**，且
ankle 排在 knee_up 之前；``Entity.actuator_names`` 是
``right 髋×4, left 髋×4, right_ankle, left_ankle``。所以本脚本所有名字表都在
运行时从 env 取真值并断言，绝不硬编码下标。

用法::

    MUJOCO_GL=egl uv run python scripts/diag_v031_rollout.py \
        --checkpoint logs/rsl_rl/pikachu_v031_velocity/<run>/model_29999.pt \
        --out /tmp/v031_diag
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

# 期望的关节顺序（与 asset XML 的 joint 定义顺序一致，right 在前）。
EXPECTED_JOINT_NAMES = (
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_joint",
  "right_knee_up_joint",
  "right_knee_mid_joint",
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_joint",
  "left_knee_up_joint",
  "left_knee_mid_joint",
)
# 期望的 actuator（动作）顺序。
EXPECTED_ACTUATOR_NAMES = (
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_up_joint",
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_up_joint",
  "right_ankle_joint",
  "left_ankle_joint",
)
# 镜像符号（+1 同号 / -1 反号），按关节基名。
MIRROR_SIGN = {
  "hip_pitch": -1.0,
  "hip_roll": -1.0,
  "hip_yaw": -1.0,
  "knee": -1.0,
  "knee_up": +1.0,
  "knee_mid": +1.0,
  "ankle": -1.0,
}


def _partner(name: str) -> str:
  # 注意 "left_" 是 5 字符而 "right_" 是 6 字符，不能共用一个切片下标。
  prefix, rest = name.split("_", 1)
  if prefix == "left":
    return "right_" + rest
  if prefix == "right":
    return "left_" + rest
  raise ValueError(f"不是左右成对的名字: {name}")


def _sign(name: str) -> float:
  base = name.split("_", 1)[1].removesuffix("_joint")
  return MIRROR_SIGN[base]


def mirror_vector(v: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
  """按名字配对做矢状面镜像：new[i] = sign(name_i) * old[partner(name_i)]。"""
  idx = {n: i for i, n in enumerate(names)}
  out = np.empty_like(np.asarray(v, dtype=np.float64))
  for i, n in enumerate(names):
    out[i] = _sign(n) * v[idx[_partner(n)]]
  return out


def quat_to_rpy(q: np.ndarray) -> np.ndarray:
  """wxyz 四元数 -> (roll, pitch, yaw)。

  ``EntityData.root_link_quat_w`` 直接来自 MuJoCo 的 qpos[3:7]，是 **wxyz** 序
  （与对外 CSV/关键帧接口的 xyzw 不同）。
  """
  w, x, y, z = q
  roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
  yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  return np.array([roll, pitch, yaw])


def _actor(obs):
  """RslRlVecEnvWrapper 的 obs 可能是 dict（含 actor/critic）或裸张量。"""
  return obs["actor"] if isinstance(obs, dict) else obs


def build(task: str, checkpoint: str, device: str, cfg):
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(load_rl_cfg(task)), log_dir=None, device=device)
  runner.load(
    checkpoint,
    load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False},
  )
  runner.alg.eval_mode()
  return env, runner.get_inference_policy(device=device)


def check_names(env) -> tuple[tuple[str, ...], tuple[str, ...]]:
  """从 env 取真实关节/动作名并与期望表核对。"""
  robot = env.scene["robot"]
  jn = tuple(robot.joint_names)
  an = tuple(robot.actuator_names)
  assert jn == EXPECTED_JOINT_NAMES, (
    f"关节顺序与脚本假设不符:\n  实际={jn}\n  期望={EXPECTED_JOINT_NAMES}"
  )
  assert an == EXPECTED_ACTUATOR_NAMES, (
    f"动作顺序与脚本假设不符:\n  实际={an}\n  期望={EXPECTED_ACTUATOR_NAMES}"
  )
  return jn, an


def make_cfg(task: str, cmd: tuple[float, float, float], seed: int, yaw0: float | None):
  cfg = load_env_cfg(task, play=False)
  cfg.scene.num_envs = 1
  cfg.episode_length_s = 1.0e9
  cfg.observations["actor"].enable_corruption = False
  cfg.events.pop("push_robot", None)
  cfg.curriculum.pop("command_vel", None)
  cfg.seed = seed
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = (cmd[0], cmd[0])
  twist.ranges.lin_vel_y = (cmd[1], cmd[1])
  twist.ranges.ang_vel_z = (cmd[2], cmd[2])
  twist.heading_command = False
  twist.ranges.heading = None
  for attr in (
    "rel_standing_envs",
    "rel_heading_envs",
    "rel_world_envs",
    "rel_forward_envs",
    "rel_backward_envs",
    "rel_lateral_envs",
  ):
    setattr(twist, attr, 0.0)
  if yaw0 is not None:
    cfg.events["reset_base"].params["pose_range"]["yaw"] = (yaw0, yaw0)
  return cfg


def rollout(task, checkpoint, device, cmd, seed, seconds, yaw0):
  env, policy = build(task, checkpoint, device, make_cfg(task, cmd, seed, yaw0))
  try:
    check_names(env)
    obs, _ = env.reset()
    robot = env.scene["robot"]
    step_dt = float(env.step_dt)
    steps = int(round(seconds / step_dt))
    rec: dict[str, list] = {
      k: []
      for k in (
        "t",
        "pos_w",
        "rpy",
        "lin_vel_b",
        "ang_vel_b",
        "q",
        "qd",
        "tau",
        "action",
      )
    }
    done_at = None
    for i in range(steps):
      with torch.no_grad():
        action = policy(obs)
      obs, _, term, trunc, _ = env.step(action)
      d = robot.data
      rec["t"].append(i * step_dt)
      rec["pos_w"].append(d.root_link_pos_w[0].cpu().numpy().copy())
      rec["rpy"].append(quat_to_rpy(d.root_link_quat_w[0].cpu().numpy()))
      rec["lin_vel_b"].append(d.root_link_lin_vel_b[0].cpu().numpy().copy())
      rec["ang_vel_b"].append(d.root_link_ang_vel_b[0].cpu().numpy().copy())
      rec["q"].append(d.joint_pos[0].cpu().numpy().copy())
      rec["qd"].append(d.joint_vel[0].cpu().numpy().copy())
      rec["tau"].append(
        d.qfrc_actuator[0, 6:].cpu().numpy().copy()
      )  # 跳过 free joint 的 6 dof
      rec["action"].append(action[0].detach().cpu().numpy().copy())
      if bool(term[0]) or bool(trunc[0]):
        done_at = i
        break
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["done_at"] = done_at
    out["step_dt"] = step_dt
    return out
  finally:
    env.close()


def summarize(tag: str, r: dict, jn: tuple[str, ...], an: tuple[str, ...]) -> dict:
  q, rpy, av = r["q"], r["rpy"], r["ang_vel_b"]
  iL, iR = jn.index("left_hip_roll_joint"), jn.index("right_hip_roll_joint")
  qL, qR = q[:, iL], q[:, iR]
  s = (qL - qR) / 2.0
  d = (qL + qR) / 2.0
  aL, aR = an.index("left_hip_roll_joint"), an.index("right_hip_roll_joint")
  out = {
    "tag": tag,
    "seconds": float(r["t"][-1]) if len(r["t"]) else 0.0,
    "done_at": r["done_at"],
    "roll_mean": float(rpy[:, 0].mean()),
    "roll_std": float(rpy[:, 0].std()),
    "pitch_mean": float(rpy[:, 1].mean()),
    "pitch_std": float(rpy[:, 1].std()),
    "yaw_drift": float(rpy[-1, 2] - rpy[0, 2]) if len(rpy) else 0.0,
    "yaw_rate_rms": float(np.sqrt(np.mean(av[:, 2] ** 2))),
    "pos_y_drift": float(r["pos_w"][-1, 1] - r["pos_w"][0, 1]) if len(rpy) else 0.0,
    "pos_x_travel": float(r["pos_w"][-1, 0] - r["pos_w"][0, 0]) if len(rpy) else 0.0,
    "qL_mean": float(qL.mean()),
    "qR_mean": float(qR.mean()),
    "qL_std": float(qL.std()),
    "qR_std": float(qR.std()),
    "qL_min": float(qL.min()),
    "qL_max": float(qL.max()),
    "qR_min": float(qR.min()),
    "qR_max": float(qR.max()),
    "sym_s_mean": float(s.mean()),
    "anti_d_mean": float(d.mean()),
    "anti_d_std": float(d.std()),
    "act_L_mean": float(r["action"][:, aL].mean()),
    "act_R_mean": float(r["action"][:, aR].mean()),
  }
  print(
    f"[{tag}] {out['seconds']:.1f}s "
    f"roll={out['roll_mean']:+.3f}±{out['roll_std']:.3f} "
    f"pitch={out['pitch_mean']:+.3f} yaw漂移={np.degrees(out['yaw_drift']):+7.1f}° "
    f"yaw_rate_rms={out['yaw_rate_rms']:.3f}\n"
    f"    pos: dx={out['pos_x_travel']:+.3f} dy={out['pos_y_drift']:+.3f}\n"
    f"    hip_roll  左 qL={out['qL_mean']:+.4f}±{out['qL_std']:.4f} "
    f"[{out['qL_min']:+.3f},{out['qL_max']:+.3f}]  "
    f"右 qR={out['qR_mean']:+.4f}±{out['qR_std']:.4f} "
    f"[{out['qR_min']:+.3f},{out['qR_max']:+.3f}]\n"
    f"    对称 s={out['sym_s_mean']:+.4f}  同向 d={out['anti_d_mean']:+.4f}±{out['anti_d_std']:.4f}\n"
    f"    raw action: actL={out['act_L_mean']:+.3f} actR={out['act_R_mean']:+.3f}"
  )
  return out


def mirror_equivariance(
  policy, obs: np.ndarray, device: str, jn: tuple[str, ...], an: tuple[str, ...]
) -> dict:
  """比较 policy(mirror(obs)) 与 mirror(policy(obs)) 的逐元素差。

  观测布局（actor，50 维，平地）：
    0:3 base_lin_vel(机体系) / 3:6 base_ang_vel(机体系) / 6:9 projected_gravity
    9:23 joint_pos(14) / 23:37 joint_vel(14) / 37:47 last_action(10) / 47:50 command
  """
  n = obs.shape[0]
  if n == 0:
    return {}
  mir = obs.copy()
  mir[:, 0:3] = obs[:, 0:3] * np.array([1, -1, 1])
  mir[:, 3:6] = obs[:, 3:6] * np.array([-1, 1, -1])
  mir[:, 6:9] = obs[:, 6:9] * np.array([1, -1, 1])
  for s, e in ((9, 23), (23, 37)):
    mir[:, s:e] = np.stack([mirror_vector(obs[i, s:e], jn) for i in range(n)])
  mir[:, 37:47] = np.stack([mirror_vector(obs[i, 37:47], an) for i in range(n)])
  mir[:, 47:50] = obs[:, 47:50] * np.array([1, -1, -1])

  with torch.no_grad():
    a_orig = policy({"actor": torch.tensor(obs, dtype=torch.float32, device=device)})
    a_mir = policy({"actor": torch.tensor(mir, dtype=torch.float32, device=device)})
    a_orig = a_orig.cpu().numpy()
    a_mir = a_mir.cpu().numpy()
  a_orig_mirrored = np.stack([mirror_vector(a_orig[i], an) for i in range(n)])
  diff = a_mir - a_orig_mirrored
  return {
    "n_samples": int(n),
    "mean_abs_diff": float(np.abs(diff).mean()),
    "max_abs_diff": float(np.abs(diff).max()),
    "per_dim_abs_diff": np.abs(diff).mean(axis=0).tolist(),
    "act_std": float(a_orig.std()),
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="Mjlab-Velocity-Flat-Pikachu-V031-T6")
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--out", default="/tmp/v031_diag")
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--seconds", type=float, default=12.0)
  ap.add_argument("--seeds", default="42", help="逗号分隔，用于 DC 偏置的 seed 统计")
  args = ap.parse_args()

  outdir = Path(args.out)
  outdir.mkdir(parents=True, exist_ok=True)
  seeds = [int(s) for s in args.seeds.split(",")]

  results = []
  print("=" * 78)
  print("A. 固定命令 rollout（yaw0=0，nominal：无推力/无噪声/无 DR 事件随机）")
  print("=" * 78)
  jn = an = None
  for cmd, name in (((0.4, 0.0, 0.0), "fwd"), ((0.0, 0.0, 0.0), "stand")):
    r = rollout(
      args.task, args.checkpoint, args.device, cmd, seeds[0], args.seconds, 0.0
    )
    if jn is None:
      env0, _ = build(
        args.task, args.checkpoint, args.device, make_cfg(args.task, cmd, seeds[0], 0.0)
      )
      try:
        jn, an = check_names(env0)
      finally:
        env0.close()
    results.append(summarize(f"{name}/seed{seeds[0]}", r, jn, an))
    if name == "fwd":
      np.savez_compressed(outdir / "rollout_fwd.npz", **{k: v for k, v in r.items()})

  print()
  print("=" * 78)
  print("B. 镜像等变性（fwd 轨迹上的全部观测）")
  print("=" * 78)
  env, policy = build(
    args.task,
    args.checkpoint,
    args.device,
    make_cfg(args.task, (0.4, 0.0, 0.0), seeds[0], 0.0),
  )
  try:
    check_names(env)
    obs, _ = env.reset()
    collected = []
    for _ in range(200):
      with torch.no_grad():
        action = policy(obs)
      collected.append(_actor(obs)[0].detach().cpu().numpy().copy())
      obs, _, term, trunc, _ = env.step(action)
      if bool(term[0]) or bool(trunc[0]):
        break
    me = mirror_equivariance(policy, np.asarray(collected), args.device, jn, an)
    print(f"样本数 = {me['n_samples']}")
    print(f"策略输出标准差 (动作尺度参考) = {me['act_std']:.4f}")
    print(f"镜像前后逐元素平均绝对差 = {me['mean_abs_diff']:.4f}")
    print(f"镜像前后逐元素最大绝对差 = {me['max_abs_diff']:.4f}")
    print("各动作维平均绝对差（按真实 actuator 顺序）:")
    for nm, v in zip(an, me["per_dim_abs_diff"], strict=True):
      print(f"    {nm:<26}{v:.4f}")
    ratio = me["mean_abs_diff"] / max(me["act_std"], 1e-9)
    print(
      f"\n等变误差 / 策略输出尺度 = {ratio:.3f}  "
      f"({'镜像等变' if ratio < 0.05 else '**非镜像等变**'})"
    )
  finally:
    env.close()

  print()
  print("=" * 78)
  print("C. 多 seed 的 DC 偏置方向（判『系统性偏置』还是『等价解』）")
  print("=" * 78)
  seed_stats = []
  for sd in seeds:
    r = rollout(args.task, args.checkpoint, args.device, (0.4, 0.0, 0.0), sd, 6.0, 0.0)
    s = summarize(f"fwd/seed{sd}", r, jn, an)
    seed_stats.append(s)
  if len(seed_stats) > 1:
    dvals = np.array([s["anti_d_mean"] for s in seed_stats])
    svals = np.array([s["sym_s_mean"] for s in seed_stats])
    print(
      f"\n同向分量 d 跨 {len(seeds)} seed: mean={dvals.mean():+.4f} "
      f"std={dvals.std():.4f}  同号比例={np.mean(np.sign(dvals) == np.sign(dvals.mean())):.2f}"
    )
    print(
      f"对称分量 s 跨 {len(seeds)} seed: mean={svals.mean():+.4f} std={svals.std():.4f}"
    )
    print("若 d 的跨 seed std 远小于 |mean|，说明是**系统性偏置**（模型/奖励造成）；")
    print("若两者同量级，说明是**对称任务的等价解随机选择**。")

  (outdir / "rollout_diag.json").write_text(
    json.dumps({"runs": results, "seeds": seed_stats}, indent=2, ensure_ascii=False)
  )
  print(f"\n结果已写出 -> {outdir}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
