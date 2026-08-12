"""录出的参考动作（CSV/npz）回放预览。

把 ``mocap.record_cli`` 录出的 CSV，或 ``scripts.csv_to_npz_pikachu``
转换出的 npz，在原生 MuJoCo viewer 里**运动学回放**（按帧直接显示
录出的 root 姿态 + 关节角），用于**检查录制/转换结果**。

注意：录出的 CSV / 转换出的 npz 都是**运动学参考序列**——npz 尤其只是
tracking 训练喂给策略的"动作参考"，与动力学无关，回放它只需要按帧写入
显示，**不能用 PD 打点去"复现"**。验证关键帧动作的动力学可行性（机器人
实际能否跟出来），请在**录制前**用 ``preview_cli --mode pd``。

交互：**空格** 暂停/继续，关闭窗口退出，``--loop`` 循环播放。COM 投影
叠加默认开启（同 ``preview``）。

支持的输入：

  - ``*.csv``：mocap 录制格式（21 列 = root_pos 3 + root_quat **xyzw** 4
    + dof_pos 14），需用 ``--input-fps`` 给出实际帧率（录制时打印，例如
    28.57）。
  - ``*.npz``：``csv_to_npz_pikachu`` 输出（``joint_pos`` / ``body_pos_w`` /
    ``body_quat_w`` / ``fps``），帧率自带，无需 ``--input-fps``。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import mujoco
import mujoco.viewer
import numpy as np

from mjlab.mocap.mocap_env import MocapEnv
from mjlab.mocap.preview import (
  _KEY_SPACE,
  _com_projection_overlay,
  _steps_per_frame,
)


class ReplayMotion:
  """一段录出的参考动作（逐帧 root + 关节角）。

  Attributes:
    joint_pos: ``(B, 14)`` 关节角（弧度），**joint_names 序**。
    root_pos: ``(B, 3)`` base_link 世界系位置。
    root_quat_xyzw: ``(B, 4)`` base_link 世界系姿态（**xyzw 序**）。
    fps: 动作帧率（Hz）。
    source: 来源路径，仅用于打印。
    kind: 来源格式，``"csv"``（录制产物）或 ``"npz"``（纯运动学参考）。
      pd 模式只对 ``"csv"`` 有意义；``"npz"`` 会被强制 kinematic。
  """

  def __init__(
    self,
    joint_pos: np.ndarray,
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    fps: float,
    source: str,
    kind: Literal["csv", "npz"],
  ):
    if joint_pos.ndim != 2 or joint_pos.shape[1] != 14:
      raise ValueError(f"joint_pos 形状应为 (B, 14)，收到 {joint_pos.shape}")
    if not (joint_pos.shape[0] == root_pos.shape[0] == root_quat_xyzw.shape[0]):
      raise ValueError("joint_pos / root_pos / root_quat_xyzw 帧数不一致")
    self.joint_pos = joint_pos.astype(np.float32)
    self.root_pos = root_pos.astype(np.float32)
    self.root_quat_xyzw = root_quat_xyzw.astype(np.float32)
    self.fps = float(fps)
    self.source = source
    self.kind = kind

  @property
  def num_frames(self) -> int:
    return self.joint_pos.shape[0]

  @property
  def duration_s(self) -> float:
    return (self.num_frames - 1) / self.fps if self.num_frames > 1 else 0.0


def load_motion(input_file: str, input_fps: float | None = None) -> ReplayMotion:
  """从 CSV 或 npz 加载一段参考动作。

  Args:
    input_file: ``*.csv``（mocap 录制格式）或 ``*.npz``
      （csv_to_npz_pikachu 输出）。按后缀自动判别。
    input_fps: 仅 CSV 需要——录制的**实际**帧率（录制结束会打印，例如
      28.57）。npz 自带 ``fps`` 字段，忽略本参数。

  Raises:
    ValueError: 输入格式无法识别，或列数/形状不符合约定。
  """
  path = Path(input_file)
  if path.suffix.lower() == ".csv":
    if input_fps is None:
      print(
        "[replay] CSV 不带帧率信息，--input-fps 默认按 30.0；"
        "实际帧率以录制时打印为准（通常约 28.57）。"
      )
      input_fps = 30.0
    motion = np.loadtxt(input_file, delimiter=",")
    if motion.ndim != 2 or motion.shape[1] != 21:
      raise ValueError(
        f"CSV 应为 21 列（root_pos 3 + root_quat xyzw 4 + dof 14），收到 {motion.shape}"
      )
    return ReplayMotion(
      joint_pos=motion[:, 7:],
      root_pos=motion[:, :3],
      root_quat_xyzw=motion[:, 3:7],
      fps=input_fps,
      source=input_file,
      kind="csv",
    )
  if path.suffix.lower() == ".npz":
    d = np.load(input_file)
    fps = float(np.asarray(d["fps"]).reshape(-1)[0])
    # body 0 = base_link；body_quat_w 是 MuJoCo 原生 wxyz 序，转成 xyzw。
    return ReplayMotion(
      joint_pos=d["joint_pos"],
      root_pos=d["body_pos_w"][:, 0, :3],
      root_quat_xyzw=d["body_quat_w"][:, 0][..., [1, 2, 3, 0]],
      fps=fps,
      source=input_file,
      kind="npz",
    )
  raise ValueError(
    f"无法识别的输入格式 '{path.suffix}'；仅支持 .csv（mocap 录制）和 "
    f".npz（csv_to_npz_pikachu 输出）。"
  )


def replay_motion(
  motion: ReplayMotion,
  mode: Literal["kinematic", "pd"] = "kinematic",
  loop: bool = False,
  fast: bool = False,
  device: str = "cpu",
  physics_dt: float = 0.005,
  write_root: bool = False,
  show_com: bool = True,
) -> None:
  """在交互 viewer 中回放一段录出的动作。

  Args:
    motion: 要回放的参考动作（见 ``ReplayMotion`` / ``load_motion``）。
    mode: ``"kinematic"``（默认，直接写参考 root+关节角）或 ``"pd"``
      （仅对 **CSV 录制产物**有意义：近似重放录制时的 PD 动力学过程。
      对 npz（纯运动学参考）会被强制 kinematic）。
    loop: 播完是否循环。False 时播完停在末帧（关闭窗口退出）。
    fast: 满速播放（不做实时节流）。pd 模式每帧本身较慢，建议配合使用。
    device: 仿真设备，``"cpu"`` 或 ``"cuda:0"``。
    physics_dt: 物理仿真步长（秒），仅 pd 模式有意义。
    write_root: 仅 CSV 的 pd 模式有意义——是否把录出的 root 姿态写回
      （钉在录制轨迹上）。默认 False，与录制默认一致：root 由浮动基座
      动力学自由演化。
    show_com: 是否叠加显示整机质心 COM 及其地面投影（黄竖线 + 绿点 +
      红点）。默认 True。

  Raises:
    RuntimeError: 当前环境无法打开原生 MuJoCo viewer（需要本地图形界面）。
  """
  if mode == "pd" and motion.kind == "npz":
    print(
      "[replay] npz 是纯运动学参考序列（csv_to_npz 细化产物，tracking 的"
      "动作参考），与动力学无关，PD 打点重放无意义；已强制 kinematic "
      "直接显示参考轨迹。验证关键帧动作的动力学可行性请用录制前的 "
      "preview_cli --mode pd。"
    )
    mode = "kinematic"

  env = MocapEnv(device=device, physics_dt=physics_dt)

  # 起始姿态：运动学写入第一帧，viewer 打开即可见。
  env.reset(
    joint_pos=motion.joint_pos[0],
    root_pos=motion.root_pos[0],
    root_quat_xyzw=motion.root_quat_xyzw[0],
  )
  env.sync_to_mj_data()

  steps_per_frame = _steps_per_frame(motion.fps, physics_dt)
  frame_interval = 1.0 / motion.fps if motion.fps > 0 else 0.0

  if mode == "pd":

    def apply(frame: int) -> None:
      env.pd_step(
        joint_target=motion.joint_pos[frame],
        root_pos=motion.root_pos[frame],
        root_quat_xyzw=motion.root_quat_xyzw[frame],
        n_steps=steps_per_frame,
        write_root=write_root,
      )
      env.sync_to_mj_data()
  else:

    def apply(frame: int) -> None:
      env.reset(
        joint_pos=motion.joint_pos[frame],
        root_pos=motion.root_pos[frame],
        root_quat_xyzw=motion.root_quat_xyzw[frame],
      )
      env.sync_to_mj_data()

  paused: list[bool] = [False]

  def on_key(keycode: int) -> None:
    if keycode == _KEY_SPACE:
      paused[0] = not paused[0]
      print("[replay] 已暂停（空格继续）" if paused[0] else "[replay] 继续播放")

  try:
    viewer = mujoco.viewer.launch_passive(
      env.mj_model,
      env.mj_data,
      key_callback=on_key,
      show_left_ui=True,
      show_right_ui=True,
    )
  except Exception as e:  # noqa: BLE001 - 需要把底层失败转成可读提示
    raise RuntimeError(
      "无法打开原生 MuJoCo viewer（需要本地图形界面）。"
      "若在无显示环境，可用 --mode kinematic 配合 --render 方案代替。"
    ) from e

  print(
    f"[replay] 回放 {motion.num_frames} 帧 (mode={mode}, "
    f"{motion.fps:.1f}fps, {motion.duration_s:.2f}s, src={motion.source})。"
    f"空格=暂停/继续，关闭窗口=退出{'，播完自动循环' if loop else ''}。"
  )

  frame = 0
  try:
    while viewer.is_running():
      if paused[0]:
        if show_com:
          _com_projection_overlay(viewer, env.mj_model, env.mj_data)
        viewer.sync()
        time.sleep(0.03)
        continue
      if frame >= motion.num_frames:
        if not loop:
          break
        frame = 0
      apply(frame)
      if show_com:
        _com_projection_overlay(viewer, env.mj_model, env.mj_data)
      viewer.sync()
      frame += 1
      if not fast and frame_interval > 0:
        time.sleep(frame_interval)
  finally:
    viewer.close()
