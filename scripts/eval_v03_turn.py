"""Pikachu v03 转向能力的**分布内**测试（纯数值，不读图）。

## 为什么需要这个脚本

``eval_v03_velocity.py`` 的 ``turn`` 档给的是**恒定角速度** 0.30 rad/s，
而训练配置是 ``heading_command=True`` —— 策略在训练里看到的是

    ang_vel_z = clip(heading_control_stiffness(0.5) × heading_error, ±0.3)

即一个**朝向位置**任务（转到目标朝向就停），**从来没见过「一直转」这种命令**。
拿恒定角速度去测，测的是分布外的输入，结论会假。

这个脚本按训练的语义测：**给一个固定的目标朝向，看机器人能不能转过去**。
判据是「8 秒后的朝向误差」和「转过多少度」，不是瞬时角速度。

同时给两个场景：
  - ``turn_only``：纯转向（线速度命令 0）—— 训练里其实没有纯转向环境，
    但这是「原地转身」最直观的问法；
  - ``walk_turn``：边走边转（vx = 0.30）—— 这才是训练里 heading 环境的真实形态。

用法::

    cd mjlab
    MUJOCO_GL=egl uv run python scripts/eval_v03_turn.py \
        --checkpoint logs/rsl_rl/pikachu_v03_velocity/<run>/model_29999.pt
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

# 场景名 -> (lin_vel_x, 目标朝向增量 rad)
SCENES: dict[str, tuple[float, float]] = {
  "turn_only": (0.0, np.pi / 2),  # 原地转 90°
  "walk_turn": (0.30, np.pi / 2),  # 边走边转 90°
  "turn_180": (0.0, np.pi),  # 原地转 180°
}


def build(task: str, checkpoint: str, device: str, env_cfg):
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(load_rl_cfg(task)), log_dir=None, device=device)
  runner.load(
    checkpoint,
    load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False},
  )
  runner.alg.eval_mode()
  return env, runner.get_inference_policy(device=device)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="Mjlab-Velocity-Flat-Pikachu-V03")
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--scenes", default="turn_only,walk_turn,turn_180")
  ap.add_argument("--seconds", type=float, default=8.0)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--seed", type=int, default=42)
  args = ap.parse_args()

  want = [s.strip() for s in args.scenes.split(",") if s.strip()]
  print(f"checkpoint: {args.checkpoint}")
  print()
  print(
    f"{'场景':<10} {'命令vx':>7} {'目标°':>7} | "
    f"{'初始误差°':>9} {'残留°':>7} {'转过°':>8} {'到位步':>7} {'收敛':>7} | "
    f"{'roll°':>6} {'pitch°':>7} {'终止':>5}",
    flush=True,
  )

  for name in want:
    vx, dpsi = SCENES[name]
    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 1e9
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    # ⚠️ 必须弹掉 command_vel 课程项 —— 它每步覆写 cfg.ranges.*（见 04-worklog）。
    cfg.curriculum.pop("command_vel", None)
    cfg.seed = args.seed

    twist = cfg.commands["twist"]
    twist.heading_command = True
    twist.ranges.lin_vel_x = (vx, vx)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (-0.3, 0.3)  # 与训练一致，heading 控制器会 clip 到这里
    # ⚠️ `ranges.heading` 是**世界系绝对朝向**，不是相对量。而
    # `velocity_env_cfg` 的 `reset_base` 把初始 yaw 随机化在 (-π, π) ——
    # 不把它钉住的话，「机器人初始朝向」与「我设的目标」毫无关系，
    # 量出来的「转过多少度」是噪声（第一版就栽在这里：r04 显示「转错方向」，
    # 其实只是它从 -120° 转到 +90°，看起来像倒着转）。
    cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)
    twist.ranges.heading = (float(dpsi), float(dpsi))
    for attr, val in (
      ("rel_standing_envs", 0.0),
      ("rel_heading_envs", 1.0),  # 唯一的环境 → 必定是 heading 环境
      ("rel_world_envs", 0.0),
      ("rel_forward_envs", 0.0),
      ("rel_backward_envs", 0.0),
      ("rel_lateral_envs", 0.0),
    ):
      setattr(twist, attr, val)

    env, policy = build(args.task, args.checkpoint, args.device, cfg)
    robot = env.scene["robot"]
    cmd_term = env.command_manager.get_term("twist")
    step_dt = float(env.step_dt)
    obs, _ = env.reset()

    def yaw_of() -> float:
      q = robot.data.root_link_quat_w[0]
      w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
      return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    # **用 env 自己的 heading_error 判收敛**，不要自己拿 yaw0 + dpsi 算 ——
    # 后者依赖「初始朝向恰好是 0」这个假设。heading_error 是 env 算的，
    # 是唯一权威的量（`_update_command` 就是用它推 ang_vel_z 的）。
    def err_deg() -> float:
      return float(np.degrees(abs(float(cmd_term.heading_error[0]))))

    yaw0 = yaw_of()
    err0 = err_deg()

    steps = int(args.seconds / step_dt)
    n_term = 0
    turned = 0.0
    prev = yaw0
    arrive_step = -1
    for k in range(steps):
      with torch.no_grad():
        action = policy(obs)
      obs, _, term, trunc, _ = env.step(action)
      if bool(term[0]) or bool(trunc[0]):
        n_term += 1
        break
      y = yaw_of()
      d = y - prev
      turned += float(np.arctan2(np.sin(d), np.cos(d)))
      prev = y
      # 到位判据：残留朝向误差 < 10°（env 自己算的），且此前没到位过
      if arrive_step < 0 and err_deg() < 10.0:
        arrive_step = k + 1

    resid = err_deg()
    # 收敛度：残留误差从 err0 收到 resid。1.0 = 完全到位；0 = 一点没动。
    conv = 1.0 - resid / err0 if err0 > 1e-6 else float("nan")

    q = robot.data.root_link_quat_w[0]
    w, x, yy, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    roll = float(
      np.degrees(np.arctan2(2 * (w * x + yy * z), 1 - 2 * (x * x + yy * yy)))
    )
    pitch = float(np.degrees(np.arcsin(np.clip(2 * (w * yy - z * x), -1, 1))))

    print(
      f"{name:<10} {vx:>7.2f} {np.degrees(dpsi):>7.0f} | "
      f"{err0:>7.1f} {resid:>8.1f} {np.degrees(turned):>9.1f} "
      f"{arrive_step if arrive_step > 0 else -1:>7} {conv:>7.2f} | "
      f"{roll:>6.1f} {pitch:>7.1f} {n_term:>5}",
      flush=True,
    )
    env.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
