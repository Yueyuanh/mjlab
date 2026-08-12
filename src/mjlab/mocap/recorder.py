"""PD 打点 + 插值 → 录制 tracking 参考数据（CSV）。

``record_to_csv`` 是录制管线的入口：把用户设定的关键帧（打点）插值成
稠密关节目标轨迹，用 ``MocapEnv`` 做 PD 打点执行，并把每帧状态写成与
``scripts/csv_to_npz.py`` 输入完全对齐的 CSV。默认录制真实浮动基座状态；
显式 ``write_root=True`` 时保留旧的 root-guided 运动学引导路径。

CSV 输出格式（每行一帧，21 列，无表头）：
  ``root_pos(x,y,z), root_quat(x,y,z,w), dof_pos(joint_names 序 14)``
  其中 ``root_quat`` 为 **xyzw 序**（csv_to_npz 加载后转 wxyz），
  关节角单位为弧度。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mjlab.mocap.keyframes import InterpConfig, PoseKeyframe, build_trajectory
from mjlab.mocap.mocap_env import MocapEnv


@dataclass
class RecordingResult:
  """一次录制的统计信息。"""

  num_frames: int  # 录制的帧数（= CSV 行数）
  fps: float  # 实际录制帧率（受物理步长取整影响，可能略偏离目标）
  duration_s: float  # 动作总时长（秒）
  out_csv: Path  # CSV 输出路径


def _steps_per_frame(fps: float, physics_dt: float) -> int:
  """每个录制帧推进的物理步数（decimation），保证帧间隔尽量接近 1/fps。"""
  return max(1, round(1.0 / (fps * physics_dt)))


def record_to_csv(
  keyframes: list[PoseKeyframe],
  out_csv: str | Path,
  fps: float = 30.0,
  easing: str = "linear",
  device: str = "cpu",
  physics_dt: float = 0.005,
  write_root: bool = False,
  verbose: bool = True,
) -> RecordingResult:
  """录制一段 PD 打点动作到 CSV。

  Args:
    keyframes: 关键帧序列（``PoseKeyframe``）。第一个关键帧的
      ``joint_pos`` 应完整（或提供全部 14 关节）。
    out_csv: 输出 CSV 路径（21 列，见模块 docstring）。
    fps: 目标录制帧率（Hz），对应 csv_to_npz 的 ``input_fps``。
      实际帧率会因物理步长取整而略偏，以返回值中的 ``fps`` 为准。
    easing: 关键帧插值缓动，``"linear"`` 或 ``"ease_in_out"``。
    device: 仿真设备，``"cpu"`` 或 ``"cuda:0"``。
    physics_dt: 物理仿真步长（秒）。
    write_root: 是否把 root 运动学写入（钉在插值轨迹上）。False 为默认
      动态模式，root 由浮动基座动力学、接触和关节 PD 自由演化；True 为
      兼容旧数据的 guided 模式。
    verbose: 是否打印录制统计。

  Returns:
    RecordingResult：帧数 / 实际帧率 / 时长 / 输出路径。
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

  # 提醒：关键帧关节角是否超出软限位（PD 目标超限没意义）。
  env.check_joint_limits(traj.dof_pos[0], joint_names="起始关键帧")

  steps_per_frame = _steps_per_frame(fps, physics_dt)
  actual_fps = 1.0 / (steps_per_frame * physics_dt)

  # 起始姿态：运动学写入第一个关键帧（不经过 PD 过渡）。
  env.reset(
    joint_pos=traj.dof_pos[0],
    root_pos=traj.root_pos[0],
    root_quat_xyzw=traj.root_quat_xyzw[0],
  )

  rows: list[np.ndarray] = []
  for i in range(traj.num_frames):
    env.pd_step(
      joint_target=traj.dof_pos[i],
      root_pos=traj.root_pos[i],
      root_quat_xyzw=traj.root_quat_xyzw[i],
      n_steps=steps_per_frame,
      write_root=write_root,
    )
    root_pos, root_quat_xyzw, dof_pos = env.get_state()
    if write_root:
      # Guided 兼容模式：root 列保留插值目标，关节列仍读 PD 实际响应。
      row_root_pos = traj.root_pos[i]
      row_root_quat = traj.root_quat_xyzw[i]
    else:
      row_root_pos = root_pos
      row_root_quat = root_quat_xyzw
    row = np.concatenate([row_root_pos, row_root_quat, dof_pos])
    rows.append(row)

  arr = np.stack(rows)  # (T, 21)
  out_path = Path(out_csv)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(out_path, arr, delimiter=",", fmt="%.6f")

  # 校验录制出来的关节角是否全部落在软限位内。
  env.check_joint_limits(arr[:, 7:], joint_names="录制轨迹")

  duration_s = traj.t[-1]
  if verbose:
    root_z_min = float(arr[:, 2].min())
    root_z_max = float(arr[:, 2].max())
    print(
      f"[mocap] 录制完成: {arr.shape[0]} 帧, "
      f"{actual_fps:.1f} fps (目标 {fps}), {duration_s:.2f}s"
    )
    print(f"[mocap] root_z 范围: [{root_z_min:.3f}, {root_z_max:.3f}] m")
    print(f"[mocap] CSV 已保存: {out_path.resolve()}")

  return RecordingResult(
    num_frames=arr.shape[0],
    fps=actual_fps,
    duration_s=duration_s,
    out_csv=out_path,
  )


def load_keyframes_json(
  path: str | Path,
) -> tuple[list[PoseKeyframe], InterpConfig]:
  """从 JSON 文件加载关键帧与插值参数。

  JSON 结构（关键帧字段见 ``PoseKeyframe``）：

  .. code-block:: json

    {
      "fps": 30.0,
      "easing": "linear",
      "keyframes": [
        {"duration": 1.5,
         "root_pos": [0, 0, 0.14],
         "root_quat": [0, 0, 0, 1],
         "joint_pos": {"left_hip_pitch_joint": -0.3, "...": 0.0}},
        {"duration": 1.2, "...": "..."}
      ]
    }

  顶层 ``fps`` / ``easing`` 可选；``joint_pos`` 的键为 XML 关节名
  （带 ``_joint`` 后缀），数值单位弧度。

  Args:
    path: JSON 文件路径。

  Returns:
    ``(keyframes, interp_cfg)``。
  """
  with open(path, encoding="utf-8") as f:
    data = json.load(f)

  kf_dicts = data.get("keyframes")
  if not isinstance(kf_dicts, list) or len(kf_dicts) == 0:
    raise ValueError(f"JSON 缺少非空 'keyframes' 列表: {path}")

  keyframes = [PoseKeyframe(**d) for d in kf_dicts]
  cfg = InterpConfig(
    fps=float(data.get("fps", 30.0)),
    easing=str(data.get("easing", "linear")),
  )
  return keyframes, cfg
