"""Experiment-specific disturbance curricula for Pikachu V031."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from mjlab.utils.noise import UniformNoiseCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


def _stage_index(step: int, stages: list[dict[str, Any]]) -> int:
  selected = 0
  for i, stage in enumerate(stages):
    if step >= int(stage["step"]):
      selected = i
  return selected


class v031_push_curriculum:
  """Stage the existing interval push from zero through weak to full strength."""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    event_name = cfg.params["event_name"]
    self._event_cfg = env.event_manager.get_term_cfg(event_name)
    self._stages = cfg.params["stages"]
    self._active = -1

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    stages: list[dict[str, Any]],
  ) -> dict[str, torch.Tensor]:
    del env_ids, event_name, stages
    index = _stage_index(env.common_step_counter, self._stages)
    if index != self._active:
      stage = self._stages[index]
      self._event_cfg.interval_range_s = tuple(stage["interval_range_s"])
      self._event_cfg.params["velocity_range"] = stage["velocity_range"]
      self._active = index
    vrange = self._event_cfg.params["velocity_range"]
    return {
      "stage": torch.tensor(index),
      "lateral_max_m_s": torch.tensor(float(vrange["y"][1])),
      "yaw_max_rad_s": torch.tensor(float(vrange["yaw"][1])),
    }


class v031_observation_noise_curriculum:
  """Scale actor additive uniform noise without rebuilding the observation manager."""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._manager = env.observation_manager
    self._ranges = cfg.params["base_ranges"]
    self._stages = cfg.params["stages"]
    self._active = -1

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    base_ranges: dict[str, tuple[float, float]],
    stages: list[dict[str, Any]],
  ) -> dict[str, torch.Tensor]:
    del env_ids, base_ranges, stages
    index = _stage_index(env.common_step_counter, self._stages)
    scale = float(self._stages[index]["scale"])
    if index != self._active:
      for term_name, (low, high) in self._ranges.items():
        term_cfg = self._manager.get_term_cfg("actor", term_name)
        noise = term_cfg.noise
        if not isinstance(noise, UniformNoiseCfg):
          raise TypeError(
            f"actor/{term_name} must use UniformNoiseCfg, got {type(noise).__name__}"
          )
        noise.n_min = float(low) * scale
        noise.n_max = float(high) * scale
        noise._tensor_cache.clear()
      self._active = index
    return {"scale": torch.tensor(scale), "stage": torch.tensor(index)}
