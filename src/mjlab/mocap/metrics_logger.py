"""Mocap PD 录制阶段的 metrics 日志（对齐 RL tracking ``metrics_*.csv``）。

RL 侧 ``play ... --viewer viser`` 用 ``MetricsCsvRecorder`` 把 policy 驱动的
电机输出录成 ``metrics_<task>__<ckpt>_<stamp>.csv``：带表头、逐关节 位置/速度/
扭矩/相对 spec 比值/超规格标记，能被 plot_logs metrics dashboard 识别并绘图。

本模块在 mocap 的「单点位插值控制」录制阶段（``MocapEnv.pd_step`` 逐帧把插值
关节目标喂给 position PD），用**同一套列命名与表头结构**产出一份并行 metrics 日志
（保存到 ``src/mjlab/mocap/logs/metrics_*.csv``），供离线对比「RL policy 控制器
vs 单点位插值控制」两种驱动方式下、电机输出（关节侧 ``qfrc_actuator`` 扭矩等）
是否一致。

与 RL 样本的差异（详见 README「与 RL 样本的可比性 & 差异」）：
  - 省略 ``foot_slip_*`` / ``action_acc``（mocap 无接触传感器 / action manager）；
  - ``time_s`` 用帧间隔（= steps_per_frame × physics_dt）作为时间轴，而非
    RL 的 ``env.step_dt``=0.02 s 控制周期；
  - ``network`` 列填调用方传入的 ``source``（标识 mocap + 动作名）。
"""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from mjlab.mocap.keyframes import InterpConfig, PoseKeyframe, build_trajectory
from mjlab.mocap.mocap_env import MocapEnv
from mjlab.mocap.recorder import _steps_per_frame
from mjlab.tasks.pikachu_metrics import PIKACHU_JOINT_NAMES, _actuator_spec

# 表头前 5 列固定（与 RL MetricsCsvRecorder 逐字节一致）。
BASE_COLUMNS = ("sample", "sim_step", "time_s", "env_idx", "network")

# 关节名 → (velocity_limit, effort_limit)，单一来源 pikachu_metrics._actuator_spec。
_SPEC_CACHE = {name: _actuator_spec(name) for name in PIKACHU_JOINT_NAMES}


def _safe_component(name: str) -> str:
  """把任意 source 字符串 sanitize 成文件名段（与 RL 侧同风格）。"""
  cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
  return cleaned or "mocap"


def build_metric_columns() -> list[str]:
  """构造 metrics 指标列（不包含 BASE_COLUMNS）。

  顺序与 RL ``add_pikachu_telemetry_metrics`` 一致：base_height + 每关节
  position/velocity/torque/{torque,velocity}_ratio + actuator_spec_violation。
  """
  cols = ["base_height_m"]
  for joint in PIKACHU_JOINT_NAMES:
    cols.append(f"joint_position_rad/{joint}")
    cols.append(f"joint_target_rad/{joint}")
    cols.append(f"joint_velocity_rad_s/{joint}")
    cols.append(f"joint_torque_nm/{joint}")
    cols.append(f"joint_torque_ratio/{joint}")
    cols.append(f"joint_velocity_ratio/{joint}")
  cols.append("actuator_spec_violation")
  return cols


def append_step(env: MocapEnv) -> list[str]:
  """采样一帧 mocap PD 的 metrics 行（数值格式化成 str，供 csv.writer）。

  必须在 ``env.pd_step(...)`` **之后**调用，这样 ``env.robot.data`` 才反映了
  最新关节目标经动力学响应后的状态。``robot.data`` 各场形状 ``(1, 14)``（或
  ``(1, 3)`` 对 root），按 ``env.joint_names`` 序排（= PIKACHU_JOINT_NAMES 序）。

  返回 ``len(build_metric_columns())`` 个字符串，可直接拼到 BASE_COLUMNS 行尾。
  """
  pos = env.robot.data.joint_pos[0].cpu().numpy()
  vel = env.robot.data.joint_vel[0].cpu().numpy()
  torque = env.robot.data.qfrc_actuator[0].cpu().numpy()
  base_height = float(env.robot.data.root_link_pos_w[0, 2].item())

  # 关节 PD 设定值（target）：ctrl 是 actuator 槽位序，经 ``_actuator_to_joint_idx``
  # 重排回 joint 序，得到每个关节的当前设定角。
  ctrl = env.robot.data.ctrl[0].cpu().numpy()
  target = np.empty(len(env.joint_names), dtype=np.float64)
  target[env._actuator_to_joint_idx] = ctrl

  row: list[str] = [f"{base_height:.9g}"]
  any_violation = False
  for idx, joint in enumerate(env.joint_names):
    p = float(pos[idx])
    tgt = float(target[idx])
    v = float(vel[idx])
    t = float(torque[idx])
    velocity_limit, effort_limit = _SPEC_CACHE[joint]
    row.append(f"{p:.9g}")
    row.append(f"{tgt:.9g}")
    row.append(f"{v:.9g}")
    row.append(f"{t:.9g}")
    row.append(f"{abs(t) / effort_limit:.9g}")
    row.append(f"{abs(v) / velocity_limit:.9g}")
    if abs(t) > effort_limit or abs(v) > velocity_limit:
      any_violation = True
  row.append("1" if any_violation else "0")
  return row


def record_metrics_to_csv(
  keyframes: list[PoseKeyframe],
  out_dir: str | Path,
  source: str,
  fps: float = 30.0,
  easing: str = "linear",
  device: str = "cpu",
  physics_dt: float = 0.005,
  write_root: bool = False,
  verbose: bool = True,
) -> Path:
  """录制一段 mocap PD 动作的 metrics 日志到 ``out_dir/metrics_<source>_<stamp>.csv``。

  参数语义与 ``recorder.record_to_csv`` 完全一致（同一套关键帧 / 插值 / 仿真
  配置），只是按帧额外采样 motors 输出。默认 root 由浮动基座动力学自由演化，
  ``write_root=True`` 时钉在插值轨迹上（与 record_to_csv 的 guided 模式对应）。

  Args:
    keyframes: 关键帧序列（``PoseKeyframe``）。
    out_dir: metrics 日志输出目录（``src/mjlab/mocap/logs`` 之类）。
    source: 写入 ``network`` 列的标识，也是文件名的一段，例如
      ``"Pikachu-Mocap-TransitionUp__pd"``。
    fps: 目标录制帧率（Hz）。
    easing: 关键帧插值缓动，``"linear"`` 或 ``"ease_in_out"``。
    device: 仿真设备。
    physics_dt: 物理仿真步长（秒）。
    write_root: 是否把 root 运动学写入。
    verbose: 是否打印统计。

  Returns:
    写入的 metrics CSV 路径。
  """
  env = MocapEnv(device=device, physics_dt=physics_dt)
  cfg = InterpConfig(fps=fps, easing=easing)
  initial_root_pos, initial_root_quat_xyzw, _ = env.get_state()
  traj = build_trajectory(
    keyframes,
    env.joint_names,
    cfg=cfg,
    initial_root_pos=initial_root_pos,
    initial_root_quat_xyzw=initial_root_quat_xyzw,
  )
  steps_per_frame = _steps_per_frame(fps, physics_dt)
  frame_dt = steps_per_frame * physics_dt

  env.reset(
    joint_pos=traj.dof_pos[0],
    root_pos=traj.root_pos[0],
    root_quat_xyzw=traj.root_quat_xyzw[0],
  )

  cols = build_metric_columns()
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  out_path = Path(out_dir) / (
    f"metrics_{_safe_component(source)}_{stamp}_with_target.csv"
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)

  with out_path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow([*BASE_COLUMNS, *cols])
    for i in range(traj.num_frames):
      env.pd_step(
        joint_target=traj.dof_pos[i],
        root_pos=traj.root_pos[i],
        root_quat_xyzw=traj.root_quat_xyzw[i],
        n_steps=steps_per_frame,
        write_root=write_root,
      )
      sim_step = (i + 1) * steps_per_frame
      frame_time = i * frame_dt
      writer.writerow([i, sim_step, f"{frame_time:.9g}", 0, source, *append_step(env)])

  if verbose:
    duration_s = traj.t[-1]
    actual_fps = 1.0 / frame_dt
    print(
      f"[mocap] metrics 日志: {traj.num_frames} 帧, "
      f"{actual_fps:.1f} fps (目标 {fps}), {duration_s:.2f}s"
    )
    print(f"[mocap] metrics CSV: {out_path.resolve()}")
  return out_path
