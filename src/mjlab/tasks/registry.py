"""Task registry system for managing environment registration and creation."""

from copy import deepcopy
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class _TaskCfg:
  env_cfg: ManagerBasedRlEnvCfg
  play_env_cfg: ManagerBasedRlEnvCfg
  rl_cfg: RslRlBaseRunnerCfg
  runner_cls: type | None


# 任务池遮蔽层（2026-09-19 依据用户要求加入）：将非 Pikachu 的参考/示例任务从
# 任务池「注销」，只保留 Pikachu 相关任务，使 list-envs / 训练任务选择更清晰。
#
# 注销 ≠ 删除：config/mdp/runner 等训练相关代码与各 config/*/__init__.py 的注册
# 调用全部原样保留，只在本函数入口短路。恢复某个任务 = 把它的 task_id 从
# `_TASK_POOL_EXCLUDED_IDS` 移除即可，无需改动任意任务代码。
_TASK_POOL_EXCLUDED_IDS: frozenset[str] = frozenset(
  {
    # Unitree G1 / Go1 参考任务（velocity / tracking，含 rough 与 no-state-estimation 变体）
    "Mjlab-Velocity-Flat-Unitree-G1",
    "Mjlab-Velocity-Rough-Unitree-G1",
    "Mjlab-Velocity-Flat-Unitree-Go1",
    "Mjlab-Velocity-Rough-Unitree-Go1",
    "Mjlab-Tracking-Flat-Unitree-G1",
    "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
    # 上游示例任务
    "Mjlab-Cartpole-Balance",
    "Mjlab-Cartpole-Swingup",
    "Mjlab-Lift-Cube-Yam",
    "Mjlab-Lift-Cube-Yam-Rgb",
    "Mjlab-Lift-Cube-Yam-Depth",
    "Mjlab-Multi-Cube-Seg-Yam",
  }
)


# 展示顺序（2026-09-19 依据用户要求加入）：list-envs / list_tasks() 按任务迭代
# 谱系排列而非字母序。顺序来源 `docs/task_lineage.md`：velocity 族在前、tracking
# 族在后，族内按代际演进 capsule → v2.1 → v2.2 → 27dof → 27dof-YRP。未列入此表
# 的任务（例如恢复注册的 upstream 示例）自动落回字母序、排在表后。
_TASK_DISPLAY_ORDER: tuple[str, ...] = (
  # velocity（task_lineage.md §2）
  "Mjlab-Velocity-Flat-Pikachu",
  "Mjlab-Velocity-Rough-Pikachu",
  "Mjlab-Velocity-Flat-Pikachu-Quad",
  "Mjlab-Velocity-Flat-Pikachu-Quad-Bounding",
  "Mjlab-Velocity-Flat-Pikachu-Sl22-Quad",
  "Mjlab-Velocity-Flat-Pikachu-27dof",
  "Mjlab-Velocity-Flat-Pikachu-27dof-Quad",
  "Mjlab-Velocity-Flat-Pikachu-27dof-Yrp",
  # tracking（task_lineage.md §3）
  "Mjlab-Tracking-Flat-Pikachu-Down",
  "Mjlab-Tracking-Flat-Pikachu-Up",
  "Mjlab-Tracking-Flat-Pikachu-Walk",
  "Mjlab-Tracking-Flat-Pikachu-Walk-Base",
  "Mjlab-Tracking-Flat-Pikachu-Sl21-Down",
  "Mjlab-Tracking-Flat-Pikachu-Sl21-Up",
  "Mjlab-Tracking-Flat-Pikachu-Sl21-Walk",
  "Mjlab-Tracking-Flat-Pikachu-Sl22-Walk",
  "Mjlab-Tracking-Flat-Pikachu-27dof-Walk",
  "Mjlab-Tracking-Flat-Pikachu-27dof-Quad-Walk",
  "Mjlab-Tracking-Flat-Pikachu-27dof-Yrp-Walk",
  "Mjlab-Tracking-Flat-Pikachu-27dof-Yrp-Quad-Walk",
)


# Private module-level registry: task_id -> task config.
_REGISTRY: dict[str, _TaskCfg] = {}


def register_mjlab_task(
  task_id: str,
  env_cfg: ManagerBasedRlEnvCfg,
  play_env_cfg: ManagerBasedRlEnvCfg,
  rl_cfg: RslRlBaseRunnerCfg,
  runner_cls: type | None = None,
) -> None:
  """Register an environment task.

  Args:
    task_id: Unique task identifier (e.g., "Mjlab-Velocity-Rough-Unitree-Go1").
    env_cfg: Environment configuration used for training.
    play_env_cfg: Environment configuration in "play" mode.
    rl_cfg: RL runner configuration.
    runner_cls: Optional custom runner class. If None, uses OnPolicyRunner.
  """
  if task_id in _TASK_POOL_EXCLUDED_IDS:
    return
  if task_id in _REGISTRY:
    raise ValueError(f"Task '{task_id}' is already registered")
  _REGISTRY[task_id] = _TaskCfg(env_cfg, play_env_cfg, rl_cfg, runner_cls)


def list_tasks() -> list[str]:
  """List all registered task IDs in lineage display order.

  Tasks appear in ``_TASK_DISPLAY_ORDER`` first; any task not listed there
  (e.g. re-enabled upstream examples) falls back to alphabetical order at the
  end. During the Pikachu-only pool phase every registered task is in the list.
  """
  rank = {task_id: i for i, task_id in enumerate(_TASK_DISPLAY_ORDER)}
  return sorted(
    _REGISTRY.keys(), key=lambda tid: (rank.get(tid, len(_TASK_DISPLAY_ORDER)), tid)
  )


def load_env_cfg(task_name: str, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Load environment configuration for a task.

  Returns a deep copy to prevent mutation of the registered config.
  """
  return deepcopy(
    _REGISTRY[task_name].env_cfg if not play else _REGISTRY[task_name].play_env_cfg
  )


def load_rl_cfg(task_name: str) -> RslRlBaseRunnerCfg:
  """Load RL configuration for a task.

  Returns a deep copy to prevent mutation of the registered config.
  """
  return deepcopy(_REGISTRY[task_name].rl_cfg)


def load_runner_cls(task_name: str) -> type | None:
  """Load the runner class for a task.

  If None, the default OnPolicyRunner will be used.
  """
  return _REGISTRY[task_name].runner_cls
