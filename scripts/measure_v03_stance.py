"""量 pikachu_v03 的**站姿横向开度**（双脚左右间距）。

用途：判断「站立时双脚挨太近」到底差多少，并给 stance 奖励定 target。

指标一律取 **base 系 y**（不是世界系 y）—— 机器人朝向会变，世界系 y 的
左右间距会随 yaw 缩水；base 系才是「双脚相对躯干张开多少」。

同时打印左右 `hip_roll` 角（v03 限位只有 ±0.26，开度主要由它决定），
以及默认 keyframe 的参考值。

用法::

    uv run python scripts/measure_v03_stance.py \
        --checkpoint logs/rsl_rl/pikachu_v03_velocity/<run>/model_29999.pt \
        --clips stand,fwd,back,lat,turn
"""

from __future__ import annotations

import argparse
import statistics as st
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Flat-Pikachu-V03"

# 与 record_v03_video.py 同一套档位。
CLIPS: dict[str, tuple[float, float, float]] = {
  "stand": (0.0, 0.0, 0.0),
  "fwd": (0.4, 0.0, 0.0),
  "back": (-0.3, 0.0, 0.0),
  "lat": (0.0, 0.25, 0.0),
  "turn": (0.0, 0.0, 0.3),
}

# 默认站姿的关节角（KNEES_BENT_KEYFRAME），用来算「策略偏离默认多少」。
DEFAULT_JOINT_POS = {
  "right_hip_pitch_joint": -0.4272,
  "left_hip_pitch_joint": -0.4252,
  "right_hip_roll_joint": 0.0,
  "left_hip_roll_joint": 0.0,
  "right_hip_yaw_joint": 0.0,
  "left_hip_yaw_joint": 0.0,
  "right_knee_up_joint": -0.2000,
  "left_knee_up_joint": 0.2000,
  "right_ankle_joint": -0.2747,
  "left_ankle_joint": 0.2727,
}


def make_env(task: str, *, vx: float, vy: float, wz: float, seed: int = 42):
  """命令收成一个点的单环境（与 record_v03_video.py 同一套做法）。"""
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = 1
  cfg.episode_length_s = 1.0e9
  cfg.observations["actor"].enable_corruption = False
  cfg.events.pop("push_robot", None)
  # ⚠️ 课程项每步覆写 cfg.ranges.*，不弹掉的话下面设的点值会被冲回默认范围。
  cfg.curriculum.pop("command_vel", None)
  cfg.seed = seed

  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = (vx, vx)
  twist.ranges.lin_vel_y = (vy, vy)
  twist.ranges.ang_vel_z = (wz, wz)
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
  return cfg


def load_policy(task: str, checkpoint: str, device: str, wrapped):
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(
    wrapped, asdict(load_rl_cfg(task)), log_dir=None, device=device
  )
  runner.load(
    checkpoint,
    load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False},
  )
  runner.alg.eval_mode()
  return runner.get_inference_policy(device=device)


def base_frame_xy(asset, site_ids) -> torch.Tensor:
  """site 世界位置 → base 系水平分量 [B, S, 2]（只去 yaw，不消俯仰/横滚）。"""
  pos_w = asset.data.site_pos_w[:, site_ids, :]  # [B, S, 3]
  root = asset.data.root_link_pose_w  # [B, 7] = pos(3) + quat(4, wxyz)
  quat_w = root[:, 3:7]
  w, x, y, z = quat_w.unbind(-1)
  # 只保留 yaw 分量（去掉 roll/pitch），把位置转到 base 的水平朝向系。
  yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  c, s = torch.cos(-yaw), torch.sin(-yaw)
  rel = pos_w[..., :2] - root[:, None, :2]
  out_x = c[:, None] * rel[..., 0] - s[:, None] * rel[..., 1]
  out_y = s[:, None] * rel[..., 0] + c[:, None] * rel[..., 1]
  return torch.stack([out_x, out_y], dim=-1)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default=TASK)
  ap.add_argument("--checkpoint", default=None, help="不给就只报默认 keyframe")
  ap.add_argument("--clips", default="stand", help=f"逗号分隔，可选 {', '.join(CLIPS)}")
  ap.add_argument("--seconds", type=float, default=6.0)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--seed", type=int, default=42)
  args = ap.parse_args()

  want = [c.strip() for c in args.clips.split(",") if c.strip()]
  bad = [c for c in want if c not in CLIPS]
  if bad:
    raise SystemExit(f"未知档位 {bad}，可选 {list(CLIPS)}")

  # 默认 keyframe 的开度只需建一次 env 就能读到（reset 后第一步 = init_state）。
  rows: list[tuple] = []
  default_sep = None

  for name in want:
    vx, vy, wz = CLIPS[name]
    cfg = make_env(args.task, vx=vx, vy=vy, wz=wz, seed=args.seed)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env)
    asset = env.scene["robot"]
    sid, sname = asset.find_sites(("right_foot", "left_foot"))
    order = list(sname)
    ri, li = order.index("right_foot"), order.index("left_foot")

    env.reset()
    base = base_frame_xy(asset, sid)
    d0 = float(base[0, li, 1] - base[0, ri, 1])
    if default_sep is None:
      default_sep = d0
      print(f"[默认 keyframe] 双脚 base 系 y 间距 = {d0 * 100:+.2f} cm")
      for i, n in enumerate(order):
        print(f"  {n:<12} base 系 y = {float(base[0, i, 1]) * 100:+.2f} cm")

    if args.checkpoint is None:
      env.close()
      continue

    policy = load_policy(args.task, args.checkpoint, args.device, wrapped)
    obs, _ = env.reset()
    step_dt = cfg.decimation * cfg.sim.mujoco.timestep
    steps = int(args.seconds / step_dt)

    jids, jnames = asset.find_joints(tuple(DEFAULT_JOINT_POS))
    jorder = list(jnames)
    sep, joints = [], []
    with torch.no_grad():
      for _ in range(steps):
        obs, _, term, trunc, _ = env.step(policy(obs))
        b = base_frame_xy(asset, sid)
        sep.append(float(b[0, li, 1] - b[0, ri, 1]))
        joints.append([float(asset.data.joint_pos[0, i]) for i in jids])
        if bool(term[0]) or bool(trunc[0]):
          break

    mean_j = [st.mean(j[k] for j in joints) for k in range(len(jorder))]
    dev = {n: mean_j[k] - DEFAULT_JOINT_POS[n] for k, n in enumerate(jorder)}
    rows.append((name, st.mean(sep), min(sep), max(sep), st.pstdev(sep), len(sep), dev))
    env.close()

  if not rows:
    return 0

  print(f"\n默认 keyframe 开度 = {default_sep * 100:+.2f} cm\n")
  print(
    f"{'档位':<7}{'命令':<20}{'间距均值':>10}{'min':>9}{'max':>9}"
    f"{'std':>7}{'比默认窄':>10}"
  )
  for name, m, lo, hi, sd, n, _ in rows:
    vx, vy, wz = CLIPS[name]
    cmd = f"{vx:+.2f},{vy:+.2f},{wz:+.2f}"
    print(
      f"{name:<7}{cmd:<20}{m * 100:>9.2f}c{lo * 100:>8.2f}c{hi * 100:>8.2f}c"
      f"{sd * 100:>6.2f}c{(default_sep - m) * 100:>9.2f}c"
    )

  print("\n各档相对默认姿态的关节偏差（mean joint_pos − 默认，单位 rad）")
  print(f"  {'关节':<26}" + "".join(f"{r[0]:>10}" for r in rows))
  for n in DEFAULT_JOINT_POS:
    print(f"  {n:<26}" + "".join(f"{r[6][n]:>+10.4f}" for r in rows))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
