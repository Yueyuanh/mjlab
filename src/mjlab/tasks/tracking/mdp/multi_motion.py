"""多参考动作库的 ``MotionCommand`` 扩展(纯增量,不影响既有单动作任务)。

动机：tracking 的 ``MotionCommand`` 一条命令只吃**一个** npz(``motion_file``),
而 Pikachu 的 rmg 参考库是 25 条速度档 clip(``mocap/csv/RMG_AMP``)。要让**一个**
策略同时跟踪整库(而不是训 25 个专家),需要把"参考"从单条变成**库**。

做法(尽量复用基类,不复制它的状态写入逻辑)：

- ``MultiMotionLoader`` 把 N 条 npz 沿**时间轴首尾拼接**成一条长序列,同时记录
  每条 clip 的 ``[start, end)`` 边界、名字、以及从文件名解析出的命令速度
  (``pikachu_walk_0.185_0.0_0.0`` -> ``(0.185, 0.0, 0.0)``)。拼接后基类所有
  ``self.motion.X[self.time_steps]`` 的属性/奖励/指标**原样可用**,无需改动。
- ``MultiMotionCommand`` 只覆写三处：
  1. ``_sample_time_steps``：先抽 clip、再在 clip 内抽帧(而不是在全库抽帧);
  2. ``_wrapped_env_ids``：走到**本 clip 末尾**就重采样(而不是走到全库末尾);
  3. ``__init__``：把单条 loader 换成 ``MultiMotionLoader`` 并重建采样直方图。

  这三处正是基类为多动作场景预留的钩子(``_sample_time_steps`` /
  ``_wrapped_env_ids`` 由 ``commands.py`` 从 ``_resample_command`` /
  ``_update_command`` 中抽出,抽出前后单动作行为逐位一致)。

额外暴露 ``clip_ids`` / ``clip_command`` / ``phase`` 三个属性,供下游
"速度命令 -> 自动匹配参考"的学生环境使用(见 ``pikachu_lite`` 任务的蒸馏档)。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.tasks.tracking.mdp.commands import (
  MotionCommand,
  MotionCommandCfg,
  MotionLoader,
)
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# ``pikachu_walk_<vx>_<vy>_<wz>``；``pikachu_stand`` 等无速度后缀的按 (0,0,0) 处理。
_CLIP_CMD_RE = re.compile(
  r"^pikachu_walk_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$"
)


def parse_clip_command(stem: str) -> tuple[float, float, float]:
  """从 clip 文件名解析命令速度 ``(vx, vy, wz)``。

  非 ``pikachu_walk_x_y_z`` 形式(如 ``pikachu_stand``)返回 ``(0, 0, 0)``。
  """
  m = _CLIP_CMD_RE.match(stem)
  if m is None:
    return (0.0, 0.0, 0.0)
  return (float(m.group(1)), float(m.group(2)), float(m.group(3)))


class MultiMotionLoader:
  """N 条 npz 沿时间轴拼接成的"动作库",附带 clip 边界与命令速度。

  逐条用基类 ``MotionLoader`` 加载(保证 body 重排/字段语义与单动作完全一致),
  再把时间轴 ``cat`` 起来。
  """

  def __init__(
    self,
    motion_files: tuple[str, ...],
    body_indexes: torch.Tensor,
    device: str = "cpu",
    src_body_names: tuple[str, ...] | None = None,
    dst_body_names: tuple[str, ...] | None = None,
  ) -> None:
    if not motion_files:
      raise ValueError("MultiMotionLoader needs at least one motion file")

    loaders = [
      MotionLoader(
        f,
        body_indexes,
        device=device,
        src_body_names=src_body_names,
        dst_body_names=dst_body_names,
      )
      for f in motion_files
    ]

    self.clip_names: tuple[str, ...] = tuple(
      f.split("/")[-1].removesuffix(".npz") for f in motion_files
    )
    self.num_clips = len(loaders)

    # 边界/长度必须与 ``time_steps`` 同设备,否则后面按 clip 索引会直接抛
    # "indices should be either on cpu or on the same device"。
    lengths = torch.tensor(
      [m.time_step_total for m in loaders], dtype=torch.long, device=device
    )
    self.clip_starts = torch.cumsum(lengths, dim=0) - lengths
    self.clip_ends = torch.cumsum(lengths, dim=0)
    self.clip_lengths = lengths
    self.time_step_total = int(lengths.sum())

    # 命令速度表:第 i 行 = 第 i 条 clip 的 (vx, vy, wz)。
    #
    # 优先用 npz 里的 ``command`` 字段(= 该 clip **稳态段的实测速度**,由
    # ``csv_to_npz_rmgamp.py`` 写入)。文件名里的 rmg 标称值(nominal)与实测差
    # 距很大 —— 例如 ``pikachu_walk_0.37_0.0_0.0`` 实测只有 0.19 m/s,且与
    # 0.185 档逐帧几乎相同。拿标称值当命令,下游"发 0.37 却只走 0.19"。
    # 没有该字段的历史 npz 回退到文件名解析。
    commands = []
    for f, n in zip(motion_files, self.clip_names, strict=True):
      with np.load(f) as data:
        if "command" in data:
          commands.append(np.asarray(data["command"], dtype=np.float32))
          continue
      commands.append(np.array(parse_clip_command(n), dtype=np.float32))
    self.clip_command = torch.from_numpy(np.stack(commands)).to(device)

    for attr in (
      "joint_pos",
      "joint_vel",
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
    ):
      setattr(self, attr, torch.cat([getattr(m, attr) for m in loaders], dim=0))

  def clip_index_of(self, time_steps: torch.Tensor) -> torch.Tensor:
    """全局帧号 -> clip 序号(右开区间,边界帧归前一条 clip)。"""
    return torch.searchsorted(self.clip_ends, time_steps, right=True).clamp(
      max=self.num_clips - 1
    )


@dataclass
class MultiMotionCommandCfg(MotionCommandCfg):
  """多动作库版本的 ``MotionCommandCfg``。

  ``motion_files`` 非空时,``motion_file`` 只作为基类初始化的占位(取
  ``motion_files[0]``),真正加载的是整个库。
  """

  motion_files: tuple[str, ...] = ()
  """参考动作库(每个元素一个 npz)。为空则退化成基类的单动作行为。"""

  start_clip_id: int = 0
  """``sampling_mode="start"`` 时固定使用第几条 clip(play/录像用)。"""

  def build(self, env: ManagerBasedRlEnv) -> "MultiMotionCommand":
    return MultiMotionCommand(self, env)


class MultiMotionCommand(MotionCommand):
  """在整个参考库里跟踪:每 env 独立持有 (clip_id, 帧号)。"""

  cfg: MultiMotionCommandCfg  # type: ignore[assignment]

  def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRlEnv):
    # 基类 __init__ 会自己用 cfg.motion_file 造单条 MotionLoader;这里临时把
    # motion_file 指向库里第一条,让基类把"单动作"那套骨架搭完,随后整体替换。
    saved = cfg.motion_file
    if cfg.motion_files:
      cfg.motion_file = cfg.motion_files[0]
    try:
      super().__init__(cfg, env)
    finally:
      cfg.motion_file = saved

    self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    if cfg.motion_files:
      self.motion = MultiMotionLoader(
        cfg.motion_files,
        self.body_indexes,
        device=self.device,
        src_body_names=cfg.motion_body_column_names,
        dst_body_names=cfg.body_names,
      )
      self._rebuild_sampling_bins()

  # -- 动作库专属的采样直方图(基类那套按单条动作建,这里按 clip 重建) ----------

  def _rebuild_sampling_bins(self) -> None:
    step_dt = self._env.step_dt
    # 所有 clip 等长(rmg 参考均为 499 帧),取最大值建矩形直方图。
    max_len = int(self.motion.clip_lengths.max())
    self.bins_per_clip = int(max_len // (1 / step_dt)) + 1
    self.bin_count = self.bins_per_clip  # 供基类 metrics 的归一化使用
    self.bin_failed_count = torch.zeros(
      self.motion.num_clips,
      self.bins_per_clip,
      dtype=torch.float,
      device=self.device,
    )
    self._current_bin_failed = torch.zeros_like(self.bin_failed_count)
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

  def _clip_relative_bins(self, env_ids: torch.Tensor) -> torch.Tensor:
    """当前帧在**所属 clip 内**的归一化 bin 序号。"""
    starts = self.motion.clip_starts[self.clip_ids[env_ids]]
    lengths = self.motion.clip_lengths[self.clip_ids[env_ids]]
    rel = (self.time_steps[env_ids] - starts).clamp(min=0)
    return torch.clamp(
      (rel * self.bins_per_clip) // lengths.clamp(min=1),
      0,
      self.bins_per_clip - 1,
    )

  def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      bins = self._clip_relative_bins(env_ids)[episode_failed]
      clips = self.clip_ids[env_ids][episode_failed]
      flat = clips * self.bins_per_clip + bins
      counts = torch.bincount(
        flat, minlength=self.motion.num_clips * self.bins_per_clip
      )
      self._current_bin_failed[:] = counts.view(
        self.motion.num_clips, self.bins_per_clip
      ).float()

    # 逐 clip 归一化成概率(与基类同样的"非因果卷积 + 均匀兜底"配方)。
    probs = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(
      self.bins_per_clip
    )
    probs = torch.nn.functional.pad(
      probs.unsqueeze(1),
      (0, self.cfg.adaptive_kernel_size - 1),
      mode="replicate",
    )
    probs = torch.nn.functional.conv1d(probs, self.kernel.view(1, 1, -1)).squeeze(1)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-12)

    # 先抽 clip(均匀),再在该 clip 的 bin 分布里抽帧。
    n = len(env_ids)
    clips = torch.randint(0, self.motion.num_clips, (n,), device=self.device)
    self.clip_ids[env_ids] = clips
    picked = torch.multinomial(probs[clips], 1, replacement=True).squeeze(1)

    lengths = self.motion.clip_lengths[clips]
    starts = self.motion.clip_starts[clips]
    rel = (
      (picked.float() + sample_uniform(0.0, 1.0, (n,), device=self.device))
      / self.bins_per_clip
      * (lengths - 1).clamp(min=1).float()
    ).long()
    self.time_steps[env_ids] = starts + rel

    # 指标(与基类同义,只是直方图按 clip 分片)。
    p_sel = probs[clips]
    H = -(p_sel * (p_sel + 1e-12).log()).sum(dim=1)
    if self.bins_per_clip > 1:
      H = H / math.log(self.bins_per_clip)
    self.metrics["sampling_entropy"][:] = H.mean()
    pmax, imax = p_sel.max(dim=1)
    self.metrics["sampling_top1_prob"][:] = pmax.mean()
    self.metrics["sampling_top1_bin"][:] = (imax.float() / self.bins_per_clip).mean()

  def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    clips = torch.randint(0, self.motion.num_clips, (n,), device=self.device)
    self.clip_ids[env_ids] = clips
    lengths = self.motion.clip_lengths[clips]
    starts = self.motion.clip_starts[clips]
    rel = (
      torch.rand(n, device=self.device) * (lengths - 1).clamp(min=1).float()
    ).long()
    self.time_steps[env_ids] = starts + rel

    self.metrics["sampling_entropy"][:] = 1.0
    self.metrics["sampling_top1_prob"][:] = 1.0 / max(self.motion.num_clips, 1)
    self.metrics["sampling_top1_bin"][:] = 0.5

  def _sample_time_steps(self, env_ids: torch.Tensor) -> None:
    if self.cfg.sampling_mode == "start":
      self.clip_ids[env_ids] = min(
        max(self.cfg.start_clip_id, 0), self.motion.num_clips - 1
      )
      starts = self.motion.clip_starts[self.clip_ids[env_ids]]
      lengths = self.motion.clip_lengths[self.clip_ids[env_ids]]
      rel = torch.full_like(lengths, self._start_frame).clamp(
        max=(lengths - 1).clamp(min=0)
      )
      self.time_steps[env_ids] = starts + rel
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    # 把起点限制在 [clip_start + start_frame, clip_end - 1]。
    starts = self.motion.clip_starts[self.clip_ids[env_ids]]
    ends = self.motion.clip_ends[self.clip_ids[env_ids]]
    self.time_steps[env_ids] = torch.clamp(
      self.time_steps[env_ids],
      starts + self._start_frame,
      ends - 1,
    )

  def _wrapped_env_ids(self) -> torch.Tensor:
    """走到**本 clip 末尾**的 env(而不是整库末尾)。"""
    clip_end = self.motion.clip_ends[self.clip_ids]
    return torch.where(self.time_steps >= clip_end)[0]

  # -- 供下游(速度命令 -> 自动匹配)使用的派生量 ------------------------------

  @property
  def clip_command(self) -> torch.Tensor:
    """每个 env 当前跟踪的 clip 对应的命令速度 ``(vx, vy, wz)``。"""
    return self.motion.clip_command[self.clip_ids]

  @property
  def clip_name(self) -> list[str]:
    return [self.motion.clip_names[i] for i in self.clip_ids.tolist()]

  @property
  def clip_phase(self) -> torch.Tensor:
    """clip 内归一化相位 ``[0, 1)``。"""
    starts = self.motion.clip_starts[self.clip_ids]
    lengths = self.motion.clip_lengths[self.clip_ids].clamp(min=1)
    return ((self.time_steps - starts).float() / lengths.float()).clamp(0.0, 1.0)

  @property
  def clip_phase_obs(self) -> torch.Tensor:
    """相位的 (cos, sin) 编码,给策略当"我在步态哪一段"的输入。"""
    ang = 2.0 * math.pi * self.clip_phase
    return torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)


__all__ = [
  "MultiMotionCommand",
  "MultiMotionCommandCfg",
  "MultiMotionLoader",
  "parse_clip_command",
]
