"""Pikachu 数据录制：关键帧（打点）定义与插值。

本模块把用户设定的"打点"（关键帧姿态序列）插值成稠密轨迹，供
``mocap_env.MocapEnv`` 的 PD 打点执行与 CSV 录制使用。

与 ``scripts/csv_to_npz.py`` 的接口约定（务必保持一致）：
  - 关节角单位为弧度，键使用 XML 关节名（带 ``_joint`` 后缀，即
    ``robot.joint_names``），CSV 的 dof 列顺序与本模块接收的
    ``joint_names`` 参数顺序严格一致。
  - ``root_quat`` 统一使用 **xyzw** 序（与 CSV 一致）；内部插值时
    先转成 wxyz 再做 slerp。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PoseKeyframe:
  """一个姿态关键帧（"打点"）。

  Args:
    joint_pos: 关节目标角（弧度），键为 XML 关节名；未列出的关节沿用
      前一个关键帧的值（第一个关键帧缺省时用 ``build_trajectory`` 的
      ``initial_*`` 参数兜底）。
    root_pos: 世界系位置 ``(x, y, z)``（米）。None 表示沿用上一关键帧。
    root_quat: 世界系姿态四元数，**xyzw 序**。None 表示沿用上一关键帧。
    duration: 本关键帧到**下一个**关键帧的过渡时长（秒）。仅当列表里
      没有任何关键帧给出 ``time`` 时生效（相对时间模式）。
    time: 本关键帧的绝对时间戳（秒）。只要有一个关键帧给出 ``time``，
      整个列表就按绝对时间解析，此时所有 ``duration`` 都被忽略。
  """

  joint_pos: dict[str, float] | None = None
  root_pos: tuple[float, float, float] | None = None
  root_quat: tuple[float, float, float, float] | None = None
  duration: float | None = None
  time: float | None = None


@dataclass
class InterpConfig:
  """关键帧插值参数（可配置）。

  Attributes:
    fps: 录制采样率（Hz），即生成的 CSV 帧率，对应 ``csv_to_npz.py``
      的 ``input_fps`` 参数。
    easing: 缓动函数。``"linear"``（默认，与 csv_to_npz 的 lerp 一致）
      或 ``"ease_in_out"``（smoothstep，段两端速度为零、中间加速）。
    use_slerp: 四元数是否用球面插值（slerp）。True 与 csv_to_npz 的
      ``quat_slerp`` 一致；False 退化为归一化线性插值（nlerp）。
  """

  fps: float = 30.0
  easing: str = "linear"
  use_slerp: bool = True


@dataclass
class Trajectory:
  """插值后的稠密轨迹，按帧组织。

  Attributes:
    t: 每帧时间戳 ``(N,)``，单位秒。
    root_pos: 世界系位置 ``(N, 3)``，单位米。
    root_quat_xyzw: 世界系姿态四元数 ``(N, 4)``，**xyzw 序**。
    dof_pos: 关节角 ``(N, 14)``，**joint_names 序**，单位弧度。
  """

  t: np.ndarray
  root_pos: np.ndarray
  root_quat_xyzw: np.ndarray
  dof_pos: np.ndarray

  @property
  def num_frames(self) -> int:
    """轨迹帧数。"""
    return len(self.t)


def _quat_wxyz_from_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
  """xyzw 序 → wxyz 序。"""
  return q_xyzw[[3, 0, 1, 2]]


def _quat_xyzw_from_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
  """wxyz 序 → xyzw 序。"""
  return q_wxyz[[1, 2, 3, 0]]


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
  """四元数球面插值（输入输出均为 wxyz 序，已归一化）。

  处理双倍角（dot<0 时取反 q1 走短弧）与小角度（线性近似避免
  sin(θ)≈0 除零）。
  """
  q0 = q0 / np.linalg.norm(q0)
  q1 = q1 / np.linalg.norm(q1)
  dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
  if dot < 0.0:  # 取短弧
    q1 = -q1
    dot = -dot
  if dot > 0.9995:  # 夹角太小，slerp 数值不稳定，退化为线性
    q = q0 + t * (q1 - q0)
    return q / np.linalg.norm(q)
  theta = np.arccos(dot)
  sin_theta = np.sin(theta)
  return np.sin((1.0 - t) * theta) / sin_theta * q0 + np.sin(t * theta) / sin_theta * q1


def _ease(blend: float, easing: str) -> float:
  """对段内进度 ``blend ∈ [0, 1]`` 应用缓动。"""
  if easing == "linear":
    return blend
  if easing == "ease_in_out":
    return blend * blend * (3.0 - 2.0 * blend)  # smoothstep
  raise ValueError(f"未知 easing: {easing}（支持 'linear' / 'ease_in_out'）")


def _resolve_times(keyframes: list[PoseKeyframe]) -> list[float]:
  """解析每个关键帧的绝对时间戳。

  ``time``（绝对）与 ``duration``（相对到下一帧）二选一：只要列表里有
  一个关键帧给了 ``time``，就全部按绝对时间解析（缺 time 报错）；
  否则按相对时间（``t[0]=0``，``t[i+1]=t[i]+kf[i].duration``）。
  """
  use_absolute = any(kf.time is not None for kf in keyframes)
  if use_absolute:
    times: list[float] = []
    for kf in keyframes:
      if kf.time is None:
        raise ValueError("关键帧混用了 time 与 duration，请统一只用一种")
      times.append(float(kf.time))
    return times
  times = [0.0]
  for kf in keyframes[:-1]:
    times.append(times[-1] + (kf.duration if kf.duration is not None else 1.0))
  return times


def _carry_forward(
  keyframes: list[PoseKeyframe],
  joint_names: tuple[str, ...],
  initial_joint_pos: dict[str, float] | None,
  initial_root_pos: np.ndarray | None,
  initial_root_quat_xyzw: np.ndarray | None,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
  """把每个关键帧的缺省量沿序列向前填充（沿用上一个指定值）。

  Returns:
    (full_root_pos, full_root_quat_xyzw, dof_full)：长度均为
    ``len(keyframes)`` 的列表/数组。root 元素为 ``(3,)``/``(4,)`` 数组，
    ``dof_full`` 为 ``(n, 14)``（joint_names 序）。
  """
  carry_pos = (
    np.asarray(initial_root_pos, dtype=float)
    if initial_root_pos is not None
    else np.zeros(3)
  )
  carry_quat = (
    np.asarray(initial_root_quat_xyzw, dtype=float)
    if initial_root_quat_xyzw is not None
    else np.array([0.0, 0.0, 0.0, 1.0])
  )
  carry_joint: dict[str, float] = dict(initial_joint_pos or {})

  full_pos: list[np.ndarray] = []
  full_quat: list[np.ndarray] = []
  full_joint: list[dict[str, float]] = []
  for kf in keyframes:
    if kf.root_pos is not None:
      carry_pos = np.asarray(kf.root_pos, dtype=float)
    if kf.root_quat is not None:
      carry_quat = np.asarray(kf.root_quat, dtype=float)
    if kf.joint_pos is not None:
      carry_joint.update(kf.joint_pos)
    full_pos.append(carry_pos.copy())
    full_quat.append(carry_quat.copy())
    full_joint.append(dict(carry_joint))

  dof_full = np.array(
    [[d[name] for name in joint_names] for d in full_joint], dtype=float
  )
  return full_pos, full_quat, dof_full


def build_trajectory(
  keyframes: list[PoseKeyframe],
  joint_names: tuple[str, ...],
  cfg: InterpConfig | None = None,
  initial_joint_pos: dict[str, float] | None = None,
  initial_root_pos: np.ndarray | None = None,
  initial_root_quat_xyzw: np.ndarray | None = None,
) -> Trajectory:
  """把关键帧列表插值为稠密轨迹。

  Args:
    keyframes: 关键帧序列（至少 1 个）。
    joint_names: 关节名元组，决定 ``dof_pos`` 的列顺序（应与
      ``robot.joint_names`` 一致）。
    cfg: 插值配置（fps / easing / 是否 slerp）。
    initial_joint_pos: 第一个关键帧缺省关节的兜底值。
    initial_root_pos / initial_root_quat_xyzw: 第一个关键帧缺省 root
      的兜底值（四元数 xyzw 序）。

  Returns:
    Trajectory：位置/关节线性插值，四元数按配置 slerp 的稠密轨迹。

  Raises:
    ValueError: 关键帧列表为空、关节名缺失、time/duration 混用等。
  """
  if not keyframes:
    raise ValueError("关键帧列表不能为空")
  if len(joint_names) < 1:
    raise ValueError("joint_names 不能为空")
  cfg = cfg or InterpConfig()
  if cfg.fps <= 0:
    raise ValueError(f"fps 必须为正，收到 {cfg.fps}")

  times = _resolve_times(keyframes)
  full_pos, full_quat, dof_full = _carry_forward(
    keyframes,
    joint_names,
    initial_joint_pos,
    initial_root_pos,
    initial_root_quat_xyzw,
  )
  n = len(keyframes)

  # 采样：每段内均匀取点，段间不重复端点，最后补上最后一个关键帧。
  frame_times: list[float] = []
  for i in range(n - 1):
    t0, t1 = times[i], times[i + 1]
    seg_len = t1 - t0
    n_seg = max(1, round(seg_len * cfg.fps))
    for k in range(n_seg):
      frame_times.append(t0 + seg_len * k / n_seg)
  frame_times.append(times[-1])

  root_pos_out: list[np.ndarray] = []
  root_quat_out: list[np.ndarray] = []
  dof_out: list[np.ndarray] = []
  for t in frame_times:
    # 定位所在段（最后一个关键帧归入最后一段、blend=1）。
    seg = 0
    for i in range(n - 1):
      if t <= times[i + 1] + 1e-9:
        seg = i
        break
    span = times[seg + 1] - times[seg]
    blend = (t - times[seg]) / span if span > 1e-12 else 0.0
    blend = float(np.clip(blend, 0.0, 1.0))
    eased = _ease(blend, cfg.easing)

    root_pos_out.append((1.0 - eased) * full_pos[seg] + eased * full_pos[seg + 1])
    dof_out.append((1.0 - eased) * dof_full[seg] + eased * dof_full[seg + 1])

    q0 = _quat_wxyz_from_xyzw(full_quat[seg])
    q1 = _quat_wxyz_from_xyzw(full_quat[seg + 1])
    if cfg.use_slerp:
      q = _quat_slerp(q0, q1, blend)
    else:
      q = (1.0 - blend) * q0 + blend * q1
      q = q / np.linalg.norm(q)
    root_quat_out.append(_quat_xyzw_from_wxyz(q))

  return Trajectory(
    t=np.asarray(frame_times, dtype=float),
    root_pos=np.stack(root_pos_out),
    root_quat_xyzw=np.stack(root_quat_out),
    dof_pos=np.stack(dof_out),
  )
