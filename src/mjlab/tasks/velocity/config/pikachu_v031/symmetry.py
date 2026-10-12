"""V031 的矢状面镜像函数，供 rsl_rl 的 symmetry extension 使用。

策略学出**非镜像等变**的解（``scripts/diag_v031_rollout.py`` 实测等变误差 /
输出尺度 = 0.78），表现为走路时左右 ``hip_roll`` 长期不对称 + 持续偏航漂移。
奖励本身是左右对称的，所以这只是"对称任务里的某个等价解"被选中了；把镜像
等变性作为**显式约束**加进 PPO 是标准解法（Mittal et al., ICRA 2024）。

镜像面取机体矢状面（``y = 0``）。向量按极/轴矢量分类变换：

- 极矢量（线速度、重力）：``(x, y, z) -> (x, -y, z)``
- 轴矢量（角速度）：``(-x, y, -z)``
- 命令 ``(vx, vy, wz) -> (vx, -vy, -wz)``
- 左右关节交换时的符号**不是**统一的 ``-1``：见 ``MIRROR_SIGN``。
  这是 ``scripts/diag_v031_mirror_equiv.py`` 用 FK 判定出来的真值
  （混合符号表的镜像残差 0.000 mm，其余候选表 1.5~172 mm）。

**顺序陷阱**：``Entity.joint_names`` 是 right 在前，``actuator_names`` 是
``right 髋×4, left 髋×4, right_ankle, left_ankle``。所以本模块所有名字表都在
运行时从 ``env`` 的 observation manager 取 term 名与维度，按名分派规则，
绝不硬编码切片下标。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
  from rsl_rl.env import VecEnv

# 观测里 joint_pos / joint_vel 的列顺序（= Entity.joint_names）。
JOINT_NAMES = (
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_joint",
  "right_knee_up_joint",
  "right_knee_mid_joint",
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_joint",
  "left_knee_up_joint",
  "left_knee_mid_joint",
)
# 动作向量的列顺序（= Entity.actuator_names）。
ACTUATOR_NAMES = (
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_up_joint",
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_up_joint",
  "right_ankle_joint",
  "left_ankle_joint",
)
# 左右镜像时的符号：+1 同号 / -1 反号。**已用 FK 数值验证**，不要凭直觉改。
MIRROR_SIGN = {
  "hip_pitch": -1.0,
  "hip_roll": -1.0,
  "hip_yaw": -1.0,
  "knee": -1.0,
  "knee_up": +1.0,
  "knee_mid": +1.0,
  "ankle": -1.0,
}

# 每个观测 term 的镜像规则。
_TERM_RULE = {
  "base_lin_vel": "polar",
  "base_ang_vel": "axial",
  "projected_gravity": "polar",
  "joint_pos": "joints",
  "joint_vel": "joints",
  "actions": "actuators",
  "command": "command",
  "foot_height": "swap",
  "foot_air_time": "swap",
  "foot_contact": "swap",
  "foot_contact_forces": "foot_forces",
}

_POLAR = (1.0, -1.0, 1.0)
_AXIAL = (-1.0, 1.0, -1.0)
_COMMAND = (1.0, -1.0, -1.0)


def _partner(name: str) -> str:
  # "left_" 是 5 字符、"right_" 是 6 字符，不能共用切片下标。
  prefix, rest = name.split("_", 1)
  if prefix == "left":
    return "right_" + rest
  if prefix == "right":
    return "left_" + rest
  raise ValueError(f"不是左右成对的名字: {name}")


def _sign(name: str) -> float:
  return MIRROR_SIGN[name.split("_", 1)[1].removesuffix("_joint")]


def _perm_and_sign(
  names: tuple[str, ...], device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
  idx = {n: i for i, n in enumerate(names)}
  perm = torch.tensor([idx[_partner(n)] for n in names], device=device)
  sign = torch.tensor([_sign(n) for n in names], device=device, dtype=dtype)
  return perm, sign


def _mirror_joints(
  x: torch.Tensor, names: tuple[str, ...], expected_dim: int
) -> torch.Tensor:
  if x.shape[-1] != expected_dim or expected_dim != len(names):
    raise ValueError(f"关节观测维度应为 {len(names)}（名字表长度），实际 {x.shape[-1]}")
  perm, sign = _perm_and_sign(names, x.device, x.dtype)
  return x[..., perm] * sign


def _mirror_axis(x: torch.Tensor, factors: tuple[float, ...]) -> torch.Tensor:
  if x.shape[-1] != 3:
    raise ValueError(f"三维向量观测应为 3 列，实际 {x.shape[-1]}")
  f = torch.tensor(factors, device=x.device, dtype=x.dtype)
  return x * f


def _mirror_swap(x: torch.Tensor, expected_dim: int) -> torch.Tensor:
  """左右脚标量项（高度 / 滞空时间 / 接触标志）：交换两列即可。"""
  if x.shape[-1] != expected_dim or expected_dim % 2 != 0:
    raise ValueError(f"足部标量观测维度应为偶数，实际 {x.shape[-1]}")
  return torch.flip(x, dims=[-1])


def _mirror_foot_forces(x: torch.Tensor, dim: int) -> torch.Tensor:
  """接触力：先按脚交换，再对每只脚做 (x, -y, z) 的矢状面镜像。"""
  if dim % 3 != 0 or (dim // 3) % 2 != 0:
    raise ValueError(f"足部接触力维度应为 2 的倍数只脚 × 3 分量，实际 {dim}")
  n_feet = dim // 3
  v = x.reshape(*x.shape[:-1], n_feet, 3)
  v = torch.flip(v, dims=[-2])
  f = torch.tensor(_POLAR, device=x.device, dtype=x.dtype)
  return (v * f).reshape(*x.shape[:-1], dim)


def _mirror_chunk(rule: str, chunk: torch.Tensor, dim: int) -> torch.Tensor:
  if rule == "polar":
    return _mirror_axis(chunk, _POLAR)
  if rule == "axial":
    return _mirror_axis(chunk, _AXIAL)
  if rule == "command":
    return _mirror_axis(chunk, _COMMAND)
  if rule == "joints":
    return _mirror_joints(chunk, JOINT_NAMES, dim)
  if rule == "actuators":
    return _mirror_joints(chunk, ACTUATOR_NAMES, dim)
  if rule == "swap":
    return _mirror_swap(chunk, dim)
  if rule == "foot_forces":
    return _mirror_foot_forces(chunk, dim)
  raise ValueError(f"未知镜像规则 {rule}")


def _unwrapped(env) -> object:
  """兼容 ``RslRlVecEnvWrapper`` 与裸 ``ManagerBasedRlEnv`` 两种入参。"""
  return getattr(env, "unwrapped", env)


def _num_actions(env) -> int:
  n = getattr(env, "num_actions", None)
  if n is not None:
    return int(n)
  return int(_unwrapped(env).action_manager.total_action_dim)  # type: ignore[attr-defined]


def _mirror_obs_group(env: VecEnv, group: str, x: torch.Tensor) -> torch.Tensor:
  """按 observation manager 的 term 名/维度切分并逐段镜像。"""
  om = _unwrapped(env).observation_manager  # type: ignore[attr-defined]
  terms = om.active_terms[group]
  dims = om.group_obs_term_dim[group]
  total = sum(int(torch.tensor(d).prod()) for d in dims)
  if x.shape[-1] != total:
    raise ValueError(f"观测组 {group} 维度应为 {total}，实际 {x.shape[-1]}")
  pieces: list[torch.Tensor] = []
  offset = 0
  for name, d in zip(terms, dims, strict=True):
    n = int(torch.tensor(d).prod())
    chunk = x[..., offset : offset + n]
    offset += n
    rule = _TERM_RULE.get(name)
    if rule is None:
      raise KeyError(
        f"观测 term '{name}' 没有镜像规则；请在 symmetry.py 的 _TERM_RULE 里补上。"
      )
    pieces.append(_mirror_chunk(rule, chunk, n))
  return torch.cat(pieces, dim=-1)


def mirror_augmentation(
  env: VecEnv,
  obs: TensorDict | None,
  actions: torch.Tensor | None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """rsl_rl symmetry extension 的 ``data_augmentation_func``。

  返回 ``[原样本; 镜像样本]`` 拼接后的 ``(obs, actions)``。rsl_rl 约定原样本
  在前半、镜像样本在后半（``compute_loss`` 依赖这个顺序）。``obs`` 或
  ``actions`` 传 ``None`` 时对应返回值也为 ``None``。

  ``env`` 既可以是训练时的 ``RslRlVecEnvWrapper``，也可以是裸 env（离线评测
  脚本会这么传）。
  """
  obs_out: TensorDict | None = None
  if obs is not None:
    n = obs.batch_size[0]
    # 两个 stub 限制叠在一起，写法上要绕一下，运行时行为与直觉一致：
    #   1. ``TensorDict.__init__`` 的入参被约束在 ``TensorCollection`` 上，普通
    #      dict 会被类型检查判错 —— 所以走 ``from_dict``；
    #   2. ``from_dict`` 的 key 类型带 ``Divergent`` 参数，**显式注解过**的 dict
    #      会被判成不可赋值，而由字面量推断出来的不会 —— 所以 ``data`` 不加注解。
    # ``from_dict`` 与构造器语义相同：共享底层张量，不复制。
    data = {}
    for key in obs.keys():
      # TensorDict 的 key 类型是 ``str | tuple[str, ...]``（嵌套键），本任务的
      # 观测组都是扁平的，这里显式归一成 str 以便下游按名查表。
      group = key if isinstance(key, str) else "/".join(key)
      v = obs[group]
      if not isinstance(v, torch.Tensor):
        raise TypeError(f"观测组 {group} 不是单个张量，无法镜像: {type(v)}")
      data[group] = torch.cat([v, _mirror_obs_group(env, group, v)], dim=0)
    obs_out = TensorDict.from_dict(data, batch_size=[2 * n], device=obs.device)

  actions_out: torch.Tensor | None = None
  if actions is not None:
    n_act = _num_actions(env)
    if actions.shape[-1] != n_act:
      raise ValueError(f"动作维度应为 {n_act}，实际 {actions.shape[-1]}")
    perm, sign = _perm_and_sign(ACTUATOR_NAMES, actions.device, actions.dtype)
    mirrored_a = actions[..., perm] * sign
    actions_out = torch.cat([actions, mirrored_a], dim=0)

  return obs_out, actions_out
