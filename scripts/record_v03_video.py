"""Pikachu v03 无头离线录像（720P），单进程跑完所有档位。

**为什么不用 ``mjlab.scripts.play``**：``play.py`` 最后要交给一个 viewer 去跑，
而 ``viewer/base.py`` 的 ``run(num_steps=None)`` 是**无限循环**。无头环境下：

  - ``DISPLAY`` 存在 → auto 选 ``NativeMujocoViewer`` → 在桌面上真开一个 GUI
    窗口等着被关掉；
  - ``DISPLAY`` 清空 → auto 选 viser → 起一个 web 服务但**停在暂停态**，
    仿真不推进（``step_trigger=lambda step: step == 0`` 的录像因此永远不触发）。

两条路都录不出东西。这里自己拿 ``env.render()`` 取帧、``imageio`` 写 mp4。

**命令怎么固定**：不去 poke ``cmd.vel_command_b``，而是**在构造 env 之前**把
``twist`` 的 ranges 收成一个点（``lin_vel_x=(vx, vx)``），并把所有
``rel_*_envs`` 归零。这样 ``_resample_command`` 每次抽出来的就是同一个值，
观测里的 command 项也天然自洽 —— 比事后改张量干净。

    ⚠️ ``rel_forward_envs`` 必须归零：``_resample_command`` 对 forward env 会把
    x 钳到 ``min=0.3``，留着它后退档会被钳成正的。

用法::

    cd mjlab
    MUJOCO_GL=egl uv run python scripts/record_v03_video.py \
        --checkpoint logs/rsl_rl/pikachu_v03_velocity/<run>/model_5000.pt \
        --out docs/pikachu_v03/videos \
        --clips stand,fwd,back,lat,turn
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

# 档位名 -> (lin_vel_x, lin_vel_y, ang_vel_z)，单位 m/s、m/s、rad/s。
CLIPS: dict[str, tuple[float, float, float]] = {
  "stand": (0.0, 0.0, 0.0),
  "fwd": (0.4, 0.0, 0.0),
  "fwd_fast": (0.5, 0.0, 0.0),
  "back": (-0.3, 0.0, 0.0),
  "lat": (0.0, 0.25, 0.0),
  "turn": (0.0, 0.0, 0.3),
}


def build_policy(task: str, checkpoint: str, device: str, env_cfg):
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  wrapped = RslRlVecEnvWrapper(env)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  rl_cfg = load_rl_cfg(task)
  runner = runner_cls(wrapped, asdict(rl_cfg), log_dir=None, device=device)
  runner.load(
    checkpoint,
    load_cfg={
      "actor": True,
      "critic": False,
      "optimizer": False,
      "iteration": False,
    },
  )
  runner.alg.eval_mode()
  return env, runner.get_inference_policy(device=device)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="Mjlab-Velocity-Flat-Pikachu-V03")
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--out", required=True, help="mp4 输出目录")
  ap.add_argument("--tag", default="", help="文件名里的档位标记")
  ap.add_argument(
    "--clips",
    default="stand,fwd,back,lat,turn",
    help=f"逗号分隔，可选 {', '.join(CLIPS)}",
  )
  ap.add_argument("--seconds", type=float, default=6.0, help="每档录多少秒")
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=720)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument(
    "--seed",
    type=int,
    default=42,
    help="固定 base_com 等 startup 随机量，保证同一个 ckpt 的录像可复现",
  )
  args = ap.parse_args()

  import imageio.v2 as imageio

  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)

  want = [c.strip() for c in args.clips.split(",") if c.strip()]
  bad = [c for c in want if c not in CLIPS]
  if bad:
    raise SystemExit(f"未知档位 {bad}，可选 {list(CLIPS)}")

  # 一档一个 env：ranges 必须在构造 env 之前收成点，见模块 docstring。
  for name in want:
    vx, vy, wz = CLIPS[name]
    env_cfg = load_env_cfg(args.task)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 1.0e9
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.events.pop("push_robot", None)
    # ⚠️ **必须弹掉 command_vel 课程项**，否则下面设的 ranges 会被它冲掉。
    # `tasks/velocity/mdp/curriculums.py::commands_vel` 每一步都用 stage 配置
    # 覆写 `cfg.ranges.*`；stage 0 的 `step=0` 恒满足，于是构造 env 前收成点的
    # ranges 在第一次 `_resample_command` 之前就被改回默认范围 —— 六个档位会
    # 录出六段**一模一样**的片子。
    env_cfg.curriculum.pop("command_vel", None)
    env_cfg.seed = args.seed  # base_com 是 startup 事件，构造时就抽完了
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height

    twist = env_cfg.commands["twist"]
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

    # step_dt = decimation × timestep（控制周期），cfg 上没有现成属性。
    step_dt = env_cfg.decimation * env_cfg.sim.mujoco.timestep
    steps = int(args.seconds / step_dt)
    dest = out_dir / (f"{name}__{args.tag}.mp4" if args.tag else f"{name}.mp4")

    env, policy = build_policy(args.task, args.checkpoint, args.device, env_cfg)
    obs, _ = env.reset()

    t0 = time.time()
    ended_at = None
    # ⚠️ **边采边写，不要先把帧攒在内存里**。旧版用 ``imageio.mimsave(frames, ...)``，
    # 300 帧 720P = 300 × 1280 × 720 × 3 ≈ **830 MB** 常驻内存。本机 swap 已满，
    # 实测某一档的编码因此从 6 s 涨到 **517 s**（同一批其它档只要 6.4 s），
    # 而且 `t_step`/`t_render` 都是正常的 11 ms / 6 ms —— 时间全在编码那一步，
    # 只看总耗时根本定位不到。流式写入把峰值内存降到 1 帧。
    #
    # 正常基准：step ~11 ms、render ~6 ms、编码 ~10 ms/帧（合计 ~30 ms/帧）。
    writer = imageio.get_writer(
      dest,
      fps=int(1.0 / step_dt),
      codec="libx264",
      quality=8,
      macro_block_size=None,
    )
    t_step = 0.0
    t_render = 0.0
    t_write = 0.0
    n_frames = 0
    slow = 0
    try:
      for step in range(steps):
        with torch.no_grad():
          action = policy(obs)
        _a = time.time()
        obs, _, terminated, truncated, _ = env.step(action)
        _b = time.time()
        frame = np.asarray(env.render())
        _c = time.time()
        writer.append_data(frame)
        _d = time.time()
        t_step += _b - _a
        t_render += _c - _b
        t_write += _d - _c
        n_frames += 1
        if (_d - _a) > 1.0:
          slow += 1
        if bool(terminated[0]) or bool(truncated[0]):
          ended_at = step
          break
    finally:
      writer.close()

    if n_frames == 0:
      print(f"[record_v03] {name}: 没采到帧，跳过", flush=True)
      env.close()
      continue

    note = f"，第 {ended_at} 步被终止" if ended_at is not None else ""
    # 每帧平均耗时超过 0.2 s 就值得怀疑（正常 ~30 ms），标出来。
    warn = "  ⚠️ 帧耗时异常" if (t_step + t_render + t_write) / n_frames > 0.2 else ""
    print(
      f"[record_v03] ok {dest.name}  {n_frames} 帧 "
      f"{dest.stat().st_size // 1024} KiB  用时 {time.time() - t0:.1f}s{note}"
      f"  [step {t_step / n_frames * 1e3:.0f} + render {t_render / n_frames * 1e3:.0f}"
      f" + write {t_write / n_frames * 1e3:.0f} ms/帧, >1s 的帧 {slow}]{warn}",
      flush=True,
    )
    env.close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
