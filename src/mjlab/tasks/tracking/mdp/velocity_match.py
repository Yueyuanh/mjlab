"""速度命令驱动的参考匹配(蒸馏的**学生**侧命令项)。

``MultiMotionCommand`` 每个 env 随机挑一条 clip 跟踪;本模块把"挑哪条 clip"
改成**由速度命令决定**:策略收到 ``(vx, vy, wz)``,命令项在参考库里做最近邻匹配,
把该 env 的参考切到对应 clip。

这就是需求里的"发送速度控制命令自动匹配模型和数据":

  - 教师(``MultiMotionCommand``)观测里带完整参考,直接跟踪;
  - 学生(本模块)观测里**没有参考**,只有速度命令 + 本体感受,参考只用来
    决定"该走哪条运动"和产生 BC 目标(见 ``mjlab.rl.dagger_ppo``)。

命令的物理含义
--------------
命令值 = 该 clip **稳态段的实测速度**(``csv_to_npz_rmgamp.py`` 写进 npz 的
``command`` 字段),不是 rmg 的标称档位。原因见转换脚本注释:rmg 的
``pikachu_walk_0.37_0.0_0.0`` 实测只有 0.19 m/s,拿标称值当命令会"发 0.37 走 0.19"。

采样
----
每个 env 独立地:
  1. 均匀抽一条 clip(**均匀技能采样**,PHP 蒸馏要求;不用 adaptive);
  2. 命令 = 该 clip 实测速度 + 抖动,抖动半宽 = 到最近邻 clip 的轴距的一半
     (即该 clip 的 Voronoi 格半宽),保证命令仍落在自己的格子里 —— 既让策略
     见到"离格"命令而不过拟合到 25 个离散点,又不至于指到别的运动上;
  3. 在该 clip 内均匀抽帧。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.tasks.tracking.mdp.multi_motion import (
  MultiMotionCommand,
  MultiMotionCommandCfg,
)
from mjlab.utils.lab_api.math import sample_uniform


def voronoi_half_widths(library: torch.Tensor) -> torch.Tensor:
  """每条样本在每根轴上的 Voronoi 格半宽 ``(C, D)``。

  取"到同轴最近邻的绝对距离的一半"。完全重复的条目(库里有几对逐帧几乎相同的
  clip)会得到接近 0 的半宽,符合直觉 —— 它们的命令本来就该几乎一样。
  """
  out = torch.zeros_like(library)
  for axis in range(library.shape[1]):
    values = library[:, axis]
    dist = (values[:, None] - values[None, :]).abs()
    dist.fill_diagonal_(float("inf"))
    out[:, axis] = 0.5 * dist.min(dim=1).values
  return out


@dataclass
class VelocityMatchMotionCommandCfg(MultiMotionCommandCfg):
  """速度命令 -> 最近邻参考 clip 的命令配置。"""

  command_jitter: float = 1.0
  """命令抖动系数,乘以 Voronoi 格半宽。1.0 = 铺满格子;0 = 命令恒为格心。"""

  def build(self, env) -> "VelocityMatchMotionCommand":
    return VelocityMatchMotionCommand(self, env)


class VelocityMatchMotionCommand(MultiMotionCommand):
  """按速度命令选参考 clip 的多动作命令项。

  ``command`` 属性返回 ``(vx, vy, wz)``(3 维),供 ``mdp.generated_commands``
  当观测项使用 —— 这是学生唯一拿到的"任务指令"。
  """

  cfg: VelocityMatchMotionCommandCfg  # type: ignore[assignment]

  def __init__(self, cfg: VelocityMatchMotionCommandCfg, env) -> None:
    super().__init__(cfg, env)

    library = self.motion.clip_command  # (num_clips, 3) 实测稳态速度
    self._half_widths = voronoi_half_widths(library) * cfg.command_jitter
    # 匹配距离按各轴库内跨度归一化,免得量纲大的 yaw 轴主导最近邻。
    self._match_scale = (library.max(dim=0).values - library.min(dim=0).values).clamp(
      min=1e-3
    )
    self.velocity_command = torch.zeros(self.num_envs, 3, device=self.device)
    # 记录每个 env 当前命令**来源**的 clip(用于统计"命令 -> 匹配"命中率)。
    self.metrics["command_match_hit"] = torch.zeros(self.num_envs, device=self.device)

  # -- 命令接口 ---------------------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    """策略看到的速度命令 ``(vx, vy, wz)``。"""
    return self.velocity_command

  def match_clip_id(self, velocity: torch.Tensor) -> torch.Tensor:
    """速度命令 -> 最近邻 clip 序号 ``(N,)``。部署/评估时用同一套匹配。"""
    dist = (
      (
        (velocity[:, None, :] - self.motion.clip_command[None, :, :])
        / self._match_scale[None, None, :]
      )
      .pow(2)
      .sum(dim=-1)
    )
    # 并列(库里有几条命令完全相同的 clip)时随机打破,保证每条 clip 都被用到。
    dist = dist + torch.rand_like(dist) * 1e-6
    return dist.argmin(dim=1)

  # -- 采样 -------------------------------------------------------------------

  def _sample_time_steps(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if self.cfg.sampling_mode == "start":
      clips = torch.full(
        (n,),
        min(max(self.cfg.start_clip_id, 0), self.motion.num_clips - 1),
        dtype=torch.long,
        device=self.device,
      )
    else:
      # 均匀技能采样:每条 clip 等概率。命令抖动落在该 clip 的 Voronoi 格内。
      clips = torch.randint(0, self.motion.num_clips, (n,), device=self.device)

    self.clip_ids[env_ids] = clips
    center = self.motion.clip_command[clips]
    jitter = sample_uniform(-1.0, 1.0, center.shape, device=self.device)
    self.velocity_command[env_ids] = center + jitter * self._half_widths[clips]

    starts = self.motion.clip_starts[clips]
    lengths = self.motion.clip_lengths[clips]
    if self.cfg.sampling_mode == "start":
      rel = torch.full_like(lengths, self._start_frame).clamp(
        max=(lengths - 1).clamp(min=0)
      )
    else:
      rel = torch.rand(n, device=self.device) * (lengths - 1).clamp(min=1).float()
      rel = rel.long()
    self.time_steps[env_ids] = starts + rel

    self.metrics["sampling_entropy"][:] = 1.0
    self.metrics["sampling_top1_prob"][:] = 1.0 / max(self.motion.num_clips, 1)
    self.metrics["sampling_top1_bin"][:] = 0.5
    # 命令抖动在格子内,理论上必然回指同一条 clip;实测命中率当作自检指标。
    self.metrics["command_match_hit"][env_ids] = (
      self.match_clip_id(self.velocity_command[env_ids]) == clips
    ).float()


__all__ = [
  "VelocityMatchMotionCommand",
  "VelocityMatchMotionCommandCfg",
  "voronoi_half_widths",
]
