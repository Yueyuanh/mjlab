"""Pikachu v03 velocity 任务的冒烟测试（不读图，纯数值）。

在正式开训前回答四个问题：

  1. 环境能否构建（资产、传感器、奖励、指标全部解析成功）？
  2. 动作维度是不是 10（v03 只有 10 个驱动关节，没有手臂）？
  3. **模型本身站不站得住** —— 关掉 push 扰动与初始高度随机化，零动作跑 6 s，
     看 base 高度是否稳在 keyframe 附近。零动作时 PD 目标就是 keyframe 的
     默认姿态，等价于"开环站立"。若这一步就塌，训练无从谈起。
  4. 完整配置（带 push_robot 每 1~3 s 推一次）下，零动作能撑多久 ——
     这才是策略真正要解决的问题，也用来确认环境没有"开局即倒"的死局。

另外校验 IMU 帧：velocity 任务的 base_lin_vel / base_ang_vel 观测直接读
XML 里的 ``robot/imu_lin_vel``（velocimeter）与 ``robot/imu_ang_vel``（gyro）。
若 ``imu`` site 的姿态不是单位四元数，读数就落在被旋转过的坐标系里，
策略看到的"机身速度"与真实 base 系速度不符 —— 这是最难察觉的一类 bug。

用法：
    MUJOCO_GL=egl uv run python scripts/smoke_v03_velocity.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

TASK = "Mjlab-Velocity-Flat-Pikachu-V03"
NUM_ENVS = 8
STEPS = 300  # 6 s @ 50 Hz

KEYFRAME_Z = 0.26728

# 本机只有 14 GB 内存（且 swap 已满），mjlab + warp 构建第二个环境会 OOM
# 被 SIGKILL。所以分两次进程跑：`--quiet` / `--noisy`。
MODE = sys.argv[1].lstrip("-") if len(sys.argv) > 1 else "quiet"


def build_env(*, quiet: bool) -> ManagerBasedRlEnv:
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = NUM_ENVS
  if quiet:
    # 关掉所有扰动：初始高度精确、不推、脚底摩擦固定。
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.0)
    cfg.events.pop("push_robot", None)
    cfg.events.pop("foot_friction", None)
    cfg.events.pop("base_com", None)
    cfg.events.pop("encoder_bias", None)
  return ManagerBasedRlEnv(cfg, device="cuda:0")


def rpy(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
  roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1, 1))
  yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  return roll, pitch, yaw


def rollout(env: ManagerBasedRlEnv, tag: str, steps: int) -> dict:
  robot = env.scene["robot"]
  env.reset()
  zero = torch.zeros((NUM_ENVS, env.action_space.shape[-1]), device="cuda:0")
  contact = env.scene["feet_ground_contact"]

  print()
  print(f"=== {tag} ===")
  print(
    f"{'step':>6} {'base_z':>9} {'|v|':>8} {'roll':>8} {'pitch':>8} "
    f"{'knee_up_R':>10} {'knee_R':>9} {'足底力N':>9} {'nan':>5}"
  )

  nan_seen = False
  min_z = float("inf")
  fell_at = None
  kn_i = robot.joint_names.index("right_knee_up_joint")
  k_i = robot.joint_names.index("right_knee_joint")
  for s in range(steps + 1):
    base_z = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    if s % 50 == 0:
      lin = robot.data.root_link_lin_vel_b
      roll, pitch, _ = rpy(robot.data.root_link_quat_w)
      nan = bool(torch.isnan(robot.data.joint_pos).any())
      nan_seen |= nan
      min_z = min(min_z, float(base_z.min()))
      if fell_at is None and float(base_z.mean()) < 0.5 * KEYFRAME_Z:
        fell_at = s
      # force: [B, num_slots, 3]，两脚竖直方向接触力之和。
      force = contact.data.force[..., 2].sum(dim=-1)
      print(
        f"{s:>6} {float(base_z.mean()):>9.5f} {float(lin.norm(dim=1).mean()):>8.5f} "
        f"{float(roll.mean()):>8.4f} {float(pitch.mean()):>8.4f} "
        f"{float(robot.data.joint_pos[:, kn_i].mean()):>10.5f} "
        f"{float(robot.data.joint_pos[:, k_i].mean()):>9.5f} "
        f"{float(force.mean()):>9.3f} {str(nan):>5}"
      )
    if s == steps:
      break
    env.step(zero)

  term = env.termination_manager
  n_ep = NUM_ENVS * (steps + 1)
  print()
  print(f"最低 base_z : {min_z:.5f} m (keyframe {KEYFRAME_Z})")
  print(f"末步 base_z : {float(base_z.mean()):.5f} m")
  print(f"首次跌破 50% 高度: {fell_at if fell_at is not None else '未发生'}")
  print(f"出现 NaN    : {nan_seen}")
  for name in term.active_terms:
    d = term.get_term(name)
    print(f"终止 {name:<20}: {int(d.sum())} / {n_ep}")
  return {"min_z": min_z, "nan": nan_seen, "fell_at": fell_at}


def main() -> None:
  quiet_mode = MODE == "quiet"
  env = build_env(quiet=quiet_mode)
  robot = env.scene["robot"]

  print("=== 环境构建 ===")
  print(f"任务       : {TASK}")
  print(f"num_envs   : {NUM_ENVS}")
  print(f"动作维度   : {env.action_space.shape[-1]}  (期望 10)")
  print(f"观测空间   : {env.observation_space}")
  print(f"关节数     : {robot.num_joints}  (期望 14)")
  print(f"执行器数   : {len(robot.actuator_names)}  (期望 10)")
  print(f"执行器名   : {list(robot.actuator_names)}")
  passive = [n for n in robot.joint_names if n not in robot.actuator_names]
  print(f"被动关节   : {passive}")

  env.reset()
  print()
  print("=== IMU 帧一致性（imu 传感器 vs base 系真值）===")
  imu_av = env.scene["robot/imu_ang_vel"].data
  imu_lv = env.scene["robot/imu_lin_vel"].data
  d_av = float((imu_av - robot.data.root_link_ang_vel_b).abs().max())
  d_lv = float((imu_lv - robot.data.root_link_lin_vel_b).abs().max())
  print(f"  ang_vel 最大偏差: {d_av:.3e} rad/s")
  print(f"  lin_vel 最大偏差: {d_lv:.3e} m/s")
  if d_av > 1e-4 or d_lv > 1e-4:
    print("  ⚠️ imu site 姿态不是单位四元数，base 速度观测落在错误坐标系！")
  else:
    print("  ✓ imu site 为单位四元数，观测帧正确")

  if quiet_mode:
    r = rollout(env, "A. 零动作开环站立（无扰动）", STEPS)
    print()
    print("=== 判定 ===")
    if r["nan"]:
      print("✗ 出现 NaN")
    elif r["fell_at"] is not None:
      print(f"✗ 无扰动下第 {r['fell_at']} 步就塌了 —— 先修模型，别开训")
    elif r["min_z"] < 0.8 * KEYFRAME_Z:
      print(f"✗ 开环站立不稳（最低 {r['min_z']:.5f} < {0.8 * KEYFRAME_Z:.5f}）")
    else:
      print(f"✓ 开环站得住（最低 {r['min_z']:.5f} m）→ 模型没问题，可以开训")
  else:
    r = rollout(env, "B. 零动作 + 完整扰动（push 每 1~3 s）", STEPS)
    print()
    print("=== 判定 ===")
    print(f"完整扰动下零动作: 最低 {r['min_z']:.5f} m，"
          f"首次跌破半高 = {r['fell_at']}（这就是策略要解决的问题）")
  env.close()


if __name__ == "__main__":
  main()
