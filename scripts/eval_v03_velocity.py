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
    f"{'base_vx':>8} {'base_vy':>8} {'base_wz':>8} | "
    f"{'跟踪比':>7} {'方向cos':>7} {'roll°':>7} {'pitch°':>7} {'终止':>5} "
    f"{'单脚支撑':>8} {'平均脚数':>7} | "
    f"{'净位移vx':>9} {'净位移vy':>9} {'净位移wz':>9}",
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
    n_feet = []  # 每步着地的**脚**数（0/1/2），按 slot 判 > 0
    b_lin = []  # 每步 base 系线速度
    b_ang = []  # 每步 base 系角速度
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
      # ⚠️ `found` 的形状是 **(num_envs, slot)**，slot 按 `primary` 正则匹配到的
      # body 逐个排 —— v03 的 `^(left|right)_ankle_link$` 匹配 2 个 body，
      # 所以是 2 个 slot：**[0] = 左脚子树、[1] = 右脚子树**，
      # 值是该子树当前的**接触点数**（站定时实测 `[10, 10]`）。
      #
      # 所以「有没有脚着地」要按 slot 判 `> 0`，**不能** `found.sum() == 1`
      # —— 后者是「全身恰好只有 1 个接触点」，跟单脚支撑是两回事。
      # 这个 bug 一开始就写进去了，用它得出的「单脚支撑 0.76」是错的。
      n_feet.append(int((contact.data.found[0] > 0).sum()))
      b_lin.append(robot.data.root_link_lin_vel_b[0].clone())
      b_ang.append(robot.data.root_link_ang_vel_b[0].clone())

    dp = robot.data.root_link_pos_w[0, :2] - p0
    n_done = len(n_feet)
    dt = n_done * step_dt
    # 命令在 base 系；起手 yaw≈0（seed 固定），世界系与 base 系近似重合，
    # 直接用世界位移除以时间作为实测速度。
    ax, ay = float(dp[0]) / dt, float(dp[1]) / dt
    aw = float(yaw_acc) / dt

    # 但**世界系净位移只在机器人不转身时才等于 base 系前进**：实测里未受
    # 命令的 yaw 可以跑到 0.3 rad/s，8 s 就是 149°，世界位移方向随之外偏，
    # 「跟踪比」会失真。所以同时给 base 系速度的**时均** —— 它与命令同系、
    # 可直接比。步态瞬时速度有周期性起伏，但 8 s 的均值不受影响
    # （这正是「用均值不用单帧」那句话的正确用法）。
    bl = torch.stack(b_lin).mean(dim=0) if b_lin else torch.zeros(3, device=args.device)
    ba = torch.stack(b_ang).mean(dim=0) if b_ang else torch.zeros(3, device=args.device)

    roll, pitch, _ = rpy(robot.data.root_link_quat_w)
    r_deg = float(roll[0]) * 180 / np.pi
    p_deg = float(pitch[0]) * 180 / np.pi

    # 迈步判据：**单脚支撑占比**（着地脚数 == 1 的步数占比）。
    # 不要用「接触状态翻转次数」当步频 —— 胶囊脚底在软接触下 found 会抖动，
    # 站着不动也能翻出 3~4 Hz 的假步频（实测过）。着地脚数没有这个毛病：
    # 真迈步时会 1↔2 交替，站着不动时恒为 2。
    c = np.asarray(n_feet) if n_feet else np.zeros(1)
    single = float((c == 1).mean())
    mean_feet = float(c.mean())

    # 跟踪比用 **base 系时均** 算 —— 与命令同系，不受转身污染。
    cmd_mag = np.hypot(vx, vy)
    act_mag = float(torch.hypot(bl[0], bl[1]))
    ratio = act_mag / cmd_mag if cmd_mag > 1e-6 else float("nan")
    # 方向余弦：+1 = 走对方向，-1 = 反向走，0 = 垂直。命令为 0 时无意义。
    if cmd_mag > 1e-6 and act_mag > 1e-6:
      cos_dir = float((bl[0] * vx + bl[1] * vy) / (act_mag * cmd_mag))
    else:
      cos_dir = float("nan")
    print(
      f"{name:<7} {vx:>7.2f} {vy:>7.2f} {wz:>7.2f} | "
      f"{float(bl[0]):>8.3f} {float(bl[1]):>8.3f} {float(ba[2]):>8.3f} | "
      f"{ratio:>7.2f} {cos_dir:>7.2f} {r_deg:>7.1f} {p_deg:>7.1f} {n_term:>5} "
      f"{single:>8.2f} {mean_feet:>7.2f} | "
      f"{ax:>8.3f} {ay:>8.3f} {aw:>8.3f}",
      flush=True,
    )
    env.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
