"""Pure-CLI metrics recorder for RL policies (no Viser/UI needed).

用零交互的方式复刻 ``play ... --viewer viser`` 里的 metrics 录制：加载一个
policy checkpoint，在 ``env.step`` 的 rollout 循环里逐控制周期把
``metrics_manager`` 的**全部** per-step 值（含 ``joint_position_rad/*``、
``joint_target_rad/*`` 设定值、``joint_torque_nm/*`` 等）写成本地
``metrics_<tag>_<stamp>_with_target.csv``。

与 Viser ``MetricsCsvRecorder`` 逐字节相同：表头 ``sample, sim_step, time_s,
env_idx, network`` + 指标列（顺序 = ``metrics_manager.active_terms`` 注册序），
``time_s`` 用 ``env.step_dt`` 控制周期。文件名恒带 ``_with_target``（因为录制
必然包含设定值目标）。产出可直接被 plot_logs dashboard 自动发现和对比。

用法：

.. code-block:: sh

  uv run python -m mjlab.scripts.record_metrics_cli Mjlab-Tracking-Flat-Pikachu \
      --ckpt /path/to/model.pt --motion-file motions/motion.npz \
      --out-dir logs --tag TransitionUp

tracking 任务需指定参考动作（``--motion-file`` 或任务默认已配置）；velocity
等无 motion 的任务忽略该参数。
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends

# 表头前 5 列与 Viser MetricsCsvRecorder 一致。
_BASE_COLUMNS = ("sample", "sim_step", "time_s", "env_idx", "network")


@dataclass
class RecordMetricsConfig:
  """Configuration for metrics recording of an RL episode."""

  ckpt: str
  """Policy checkpoint: ``model_*.pt`` 文件，或含 ``model_*.pt`` 的目录（取最新）。"""
  motion_file: str | None = None
  """Tracking 任务参考动作 npz；缺省用任务已配置的 motion。"""
  out_dir: str = "logs"
  """metrics CSV 输出目录。"""
  tag: str | None = None
  """写入 ``network`` 列 / 文件名的标识；缺省取 ckpt 名。"""
  max_steps: int = 1000
  """单集最大控制步数上限（0.02 s/步），一整集对应约 10~20 s。"""
  sampling_mode: Literal["adaptive", "uniform", "start"] = "start"
  """tracking 参考动作取帧方式（evaluate 同款：``start`` 从参考起点）。"""
  device: str | None = None
  """运行设备；缺省 CUDA 可用则 CUDA，否则 CPU。"""
  env_idx: int = 0
  """录制这个 env 的指标（场景为单 env，固定 0）。"""


def _safe_component(name: str) -> str:
  """把任意 source 字符串 sanitize 成文件名段（与 RL 侧同风格）。"""
  cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
  return cleaned or "metrics"


def _resolve_ckpt_entry(ckpt: str) -> Path:
  """支持传文件或目录（目录取最新 ``model_*.pt``）。"""
  path = Path(ckpt)
  if not path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {path}")
  if path.is_dir():
    candidates = sorted(path.glob("model_*.pt"))
    if not candidates:
      msg = f"No model_*.pt found under {path}"
      raise FileNotFoundError(msg)
    path = candidates[-1]
  return path


def run_record(task_id: str, cfg: RecordMetricsConfig) -> Path:
  """装配环境与策略，rollout 一集并录制 metrics CSV，返回产物路径。"""
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=False)
  agent_cfg = load_rl_cfg(task_id)

  # tracking 任务：设置参考动作来源与取帧方式。
  motion = env_cfg.commands.get("motion")
  if isinstance(motion, MotionCommandCfg):
    if cfg.motion_file is not None:
      motion.motion_file = cfg.motion_file
    elif not (motion.motion_file or "").strip():
      msg = f"Task {task_id} 的 motion_file 未配置；请用 --motion-file 指定参考动作 npz"
      raise ValueError(msg)
    motion.sampling_mode = cfg.sampling_mode

  env_cfg.observations["actor"].enable_corruption = True
  env_cfg.events.pop("push_robot", None)
  env_cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  ckpt_path = _resolve_ckpt_entry(cfg.ckpt)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt_path), map_location=device)
  policy = runner.get_inference_policy(device=device)

  step_dt = float(env.unwrapped.step_dt)
  metric_names = env.unwrapped.metrics_manager.active_terms

  tag = cfg.tag or _safe_component(ckpt_path.stem)
  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  out_path = out_dir / f"metrics_{_safe_component(tag)}_{stamp}_with_target.csv"

  n_metrics = len(metric_names)
  print(f"[INFO] Recording metrics for {tag}, {n_metrics} terms, dt={step_dt}s")
  print(f"[INFO] Output: {out_path}")

  obs = env.get_observations()
  done = False
  with out_path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow([*_BASE_COLUMNS, *metric_names])
    stream.flush()
    for step in range(1, cfg.max_steps + 1):
      with torch.no_grad():
        actions = policy(obs)
      obs, _, dones, _ = env.step(actions)

      terms = dict(env.unwrapped.metrics_manager.get_active_iterable_terms(cfg.env_idx))
      row = [
        step - 1,
        step,
        f"{(step - 1) * step_dt:.9g}",
        cfg.env_idx,
        tag,
        *(terms.get(name, [float("nan")])[0] for name in metric_names),
      ]
      writer.writerow([f"{v:.9g}" if isinstance(v, float) else v for v in row])
      if step % 100 == 0:
        print(f"[INFO] step {step}/{cfg.max_steps}")
        stream.flush()

      if bool(dones.bool().all()):
        done = True
        break

  status = "episode done" if done else f"reached max_steps={cfg.max_steps}"
  print(f"[INFO] Finished ({status}), {step} steps, saved {out_path}")
  env.close()
  return out_path


def main() -> None:
  import mjlab.tasks  # noqa: F401

  tracking_tasks = [t for t in list_tasks() if "Tracking" in t]
  if not tracking_tasks:
    print("No tracking tasks found.")
    sys.exit(1)

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(tracking_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  cfg = tyro.cli(
    RecordMetricsConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_record(chosen_task, cfg)


if __name__ == "__main__":
  main()
