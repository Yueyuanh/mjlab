"""Pikachu v03 velocity 策略的定量评测（纯数值，不读图）。

回答一个看板答不了的问题：**策略到底在走，还是站着不动拿基线分？**

看板上 ``Episode_Reward/track_linear_velocity = 1.4/2.0`` 这种数字是**二义的**：
命令幅度平均 ~0.25 m/s 时，站着不动也能拿到约 0.78 的 exp 项。所以必须
直接测**净位移** —— 把命令固定成几个档位，跑固定步数，量 base 在世界系的
实际位移除以时间，与命令对比。

    ⚠️ 用净位移而不是瞬时速度：步态的瞬时速度有周期性起伏，均值可信但
    单帧不可信（见记忆 amp-v4-behavior-direction）。

同时给出：
  - 姿态健康度（roll/pitch 是否爆掉）
  - 足底交替（是否真在迈步：左右脚 air_time 交替次数）
  - 终止次数（摔没摔）

命令通过**在构造 env 之前**把 twist 的 ranges 收成一个点来固定，
与 record_v03_video.py 同一套做法。

用法::

    cd mjlab
    MUJOCO_GL=egl uv run python scripts/eval_v03_velocity.py \
        --checkpoint logs/rsl_rl/pikachu_v03_velocity/<run>/model_5000.pt
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

# 档位 -> (lin_vel_x, lin_vel_y, ang_vel_z)
CLIPS: dict[str, tuple[float, float, float]] = {
  "stand": (0.0, 0.0, 0.0),
  "fwd": (0.4, 0.0, 0.0),
  "back": (-0.3, 0.0, 0.0),
  "lat": (0.0, 0.25, 0.0),
  "turn": (0.0, 0.0, 0.3),
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


def rpy(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
  roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1, 1))
  yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  return roll, pitch, yaw


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="Mjlab-Velocity-Flat-Pikachu-V03")
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--clips", default="stand,fwd,back,lat,turn")
  ap.add_argument("--seconds", type=float, default=8.0)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--seed", type=int, default=42)
  args = ap.parse_args()

  want = [c.strip() for c in args.clips.split(",") if c.strip()]
  print(f"checkpoint: {args.checkpoint}")
  print()
  print(
    f"{'档位':<7} {'命令vx':>7} {'命令vy':>7} {'命令wz':>7} | "
    f"{'实测vx':>8} {'实测vy':>8} {'实测wz':>8} | "
    f"{'跟踪比':>7} {'roll°':>7} {'pitch°':>7} {'终止':>5} "
    f"{'单脚支撑':>8} {'平均脚数':>7}",
    flush=True,
  )

  for name in want:
    vx, vy, wz = CLIPS[name]
    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 1e9
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    # ⚠️ **必须弹掉 command_vel 课程项**，否则下面设的 ranges 会被它冲掉。
    # `tasks/velocity/mdp/curriculums.py::commands_vel` 每一步都用 stage 配置
    # 覆写 `cfg.ranges.*`；stage 0 的 `step=0` 恒满足，于是构造 env 前设的
    # `(vx, vx)` 在第一次 `_resample_command` 之前就被改回 `(-0.4, 0.5)`。
    # 症状极具迷惑性：**五个档位跑出完全相同的结果**（因为命令其实一直是
    # 同一个默认范围的随机抽样），看起来像「策略忽略命令」，实则是命令没设进去。
    cfg.curriculum.pop("command_vel", None)
    cfg.seed = args.seed
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (vx, vx)
    twist.ranges.lin_vel_y = (vy, vy)
    twist.ranges.ang_vel_z = (wz, wz)
    twist.heading_command = False
    # heading_command=False 时 ranges.heading 必须为 None，否则
    # UniformVelocityCommand.__init__ 直接抛 ValueError。
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

    env, policy = build(args.task, args.checkpoint, args.device, cfg)
    robot = env.scene["robot"]
    contact = env.scene["feet_ground_contact"]
    # step_dt 在 env 上，不在 cfg 上。
    step_dt = float(env.step_dt)
    obs, _ = env.reset()

    # 净位移用世界系位置差；yaw 用四元数解出的角差（unwrap）。
    p0 = robot.data.root_link_pos_w[0, :2].clone()
    yaw0 = rpy(robot.data.root_link_quat_w)[2][0].clone()
    yaw_acc = torch.zeros((), device=args.device)
    prev_yaw = yaw0.clone()

    steps = int(args.seconds / step_dt)
    n_term = 0
    n_contact = []  # 每步在接触中的脚数（0/1/2）
    for _ in range(steps):
      with torch.no_grad():
        action = policy(obs)
      obs, _, term, trunc, _ = env.step(action)
      if bool(term[0]) or bool(trunc[0]):
        n_term += 1
        break
      y = rpy(robot.data.root_link_quat_w)[2][0]
      d = y - prev_yaw
      yaw_acc += torch.atan2(torch.sin(d), torch.cos(d))
      prev_yaw = y
      n_contact.append(float(contact.data.found[0].sum()))

    dp = robot.data.root_link_pos_w[0, :2] - p0
    n_done = len(n_contact)
    dt = n_done * step_dt
    # 命令在 base 系；起手 yaw≈0（seed 固定），世界系与 base 系近似重合，
    # 直接用世界位移除以时间作为实测速度。
    ax, ay = float(dp[0]) / dt, float(dp[1]) / dt
    aw = float(yaw_acc) / dt

    roll, pitch, _ = rpy(robot.data.root_link_quat_w)
    r_deg = float(roll[0]) * 180 / np.pi
    p_deg = float(pitch[0]) * 180 / np.pi

    # 迈步判据：**单脚支撑占比**。
    # 不要用「接触状态翻转次数」当步频 —— 胶囊脚底在软接触下 found 会抖动，
    # 站着不动也能翻出 3~4 Hz 的假步频（实测过）。单脚支撑占比没有这个毛病：
    # 真迈步时它稳定在 0.3~0.6，站着不动时接近 0。
    c = np.asarray(n_contact) if n_contact else np.zeros(1)
    single = float((c == 1).mean())
    mean_feet = float(c.mean())

    cmd_mag = np.hypot(vx, vy)
    act_mag = np.hypot(ax, ay)
    ratio = act_mag / cmd_mag if cmd_mag > 1e-6 else float("nan")
    print(
      f"{name:<7} {vx:>7.2f} {vy:>7.2f} {wz:>7.2f} | "
      f"{ax:>8.3f} {ay:>8.3f} {aw:>8.3f} | "
      f"{ratio:>7.2f} {r_deg:>7.1f} {p_deg:>7.1f} {n_term:>5} "
      f"{single:>8.2f} {mean_feet:>7.2f}",
      flush=True,
    )
    env.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
