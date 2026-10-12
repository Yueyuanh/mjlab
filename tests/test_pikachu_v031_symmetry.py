"""V031 镜像增强函数的纯函数级测试（不建 env，快）。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from mjlab.tasks.velocity.config.pikachu_v031.symmetry import (
  ACTUATOR_NAMES,
  JOINT_NAMES,
  _mirror_chunk,
  _mirror_foot_forces,
  _mirror_joints,
  _partner,
  _sign,
  mirror_augmentation,
)


def test_partner_handles_unequal_prefix_lengths():
  # "left_" 是 5 字符、"right_" 是 6 字符：切片下标不能共用。
  assert _partner("left_hip_pitch_joint") == "right_hip_pitch_joint"
  assert _partner("right_hip_pitch_joint") == "left_hip_pitch_joint"
  assert _partner("left_knee_mid_joint") == "right_knee_mid_joint"


def test_sign_table_covers_every_joint():
  for n in JOINT_NAMES + ACTUATOR_NAMES:
    assert _sign(n) in (-1.0, 1.0)
  # 与 FK 判定出的真值表一致。
  assert _sign("left_knee_up_joint") == 1.0
  assert _sign("right_knee_mid_joint") == 1.0
  assert _sign("left_hip_roll_joint") == -1.0


@pytest.mark.parametrize("names", [JOINT_NAMES, ACTUATOR_NAMES])
def test_mirror_is_an_involution(names):
  x = torch.randn(5, len(names))
  once = _mirror_joints(x, names, len(names))
  twice = _mirror_joints(once, names, len(names))
  torch.testing.assert_close(twice, x)
  # 非平凡：确实改了东西。
  assert not torch.allclose(once, x)


def test_mirror_swaps_left_and_right_columns():
  # 只让左 hip_roll 非零，镜像后应该只有右 hip_roll 非零、且反号。
  x = torch.zeros(1, len(JOINT_NAMES))
  x[0, JOINT_NAMES.index("left_hip_roll_joint")] = 0.3
  out = _mirror_joints(x, JOINT_NAMES, len(JOINT_NAMES))
  assert out[0, JOINT_NAMES.index("right_hip_roll_joint")].item() == pytest.approx(-0.3)
  assert out.abs().sum().item() == pytest.approx(0.3)


@pytest.mark.parametrize(
  ("rule", "factors"),
  [
    ("polar", (1.0, -1.0, 1.0)),
    ("axial", (-1.0, 1.0, -1.0)),
    ("command", (1.0, -1.0, -1.0)),
  ],
)
def test_vector_rules(rule, factors):
  x = torch.randn(4, 3)
  torch.testing.assert_close(_mirror_chunk(rule, x, 3), x * torch.tensor(factors))


def test_swap_rule_flips_two_feet():
  x = torch.tensor([[1.0, 2.0]])
  torch.testing.assert_close(_mirror_chunk("swap", x, 2), torch.tensor([[2.0, 1.0]]))


def test_foot_forces_swap_feet_and_negate_y():
  # 布局是 [左脚(3), 右脚(3)]：交换两只脚 + 每只脚 (x, -y, z)。
  x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
  out = _mirror_foot_forces(x, 6)
  torch.testing.assert_close(out, torch.tensor([[4.0, -5.0, 6.0, 1.0, -2.0, 3.0]]))
  # 对合。
  torch.testing.assert_close(_mirror_foot_forces(out, 6), x)


def test_wrong_dim_raises():
  with pytest.raises(ValueError):
    _mirror_chunk("polar", torch.randn(2, 4), 4)
  # swap 只接受偶数维（两只脚）。
  with pytest.raises(ValueError):
    _mirror_chunk("swap", torch.randn(2, 3), 3)
  # joints 的维度必须与名字表长度一致。
  with pytest.raises(ValueError):
    _mirror_chunk("joints", torch.randn(2, 7), 7)
  # 未知规则在 _mirror_chunk 里直接拒绝。
  with pytest.raises(ValueError):
    _mirror_chunk("no_such_rule", torch.randn(2, 3), 3)


class _FakeObsManager:
  """只提供 ``mirror_augmentation`` 真正会读的两个字段。"""

  def __init__(self, terms: dict[str, list[tuple[str, tuple[int, ...]]]]):
    self.active_terms = {g: [n for n, _ in t] for g, t in terms.items()}
    self.group_obs_term_dim = {g: [d for _, d in t] for g, t in terms.items()}


def _fake_env(terms, num_actions: int = len(ACTUATOR_NAMES)):
  return SimpleNamespace(
    observation_manager=_FakeObsManager(terms),
    action_manager=SimpleNamespace(total_action_dim=num_actions),
  )


_TERMS = {
  "actor": [
    ("base_lin_vel", (3,)),
    ("base_ang_vel", (3,)),
    ("projected_gravity", (3,)),
    ("joint_pos", (len(JOINT_NAMES),)),
    ("joint_vel", (len(JOINT_NAMES),)),
    ("actions", (len(ACTUATOR_NAMES),)),
    ("command", (3,)),
  ],
  "critic": [
    ("foot_contact_forces", (6,)),
    ("foot_height", (2,)),
  ],
}


def test_mirror_augmentation_doubles_batch_and_is_an_involution():
  """覆盖 TensorDict 构造这条路径 —— 纯函数测试碰不到它。"""
  env = _fake_env(_TERMS)
  n = 4
  obs = TensorDict(
    {
      "actor": torch.randn(n, 50),
      "critic": torch.randn(n, 8),
    },
    batch_size=[n],
  )
  actions = torch.randn(n, len(ACTUATOR_NAMES))

  obs_out, actions_out = mirror_augmentation(env, obs, actions)
  assert obs_out is not None and actions_out is not None
  assert obs_out.batch_size == torch.Size([2 * n])
  assert actions_out.shape == (2 * n, len(ACTUATOR_NAMES))
  # 原样本必须原封不动地留在前半（rsl_rl 的 compute_loss 依赖这个顺序）。
  torch.testing.assert_close(obs_out["actor"][:n], obs["actor"])
  torch.testing.assert_close(actions_out[:n], actions)

  # 对合：对镜像样本再镜像一次，得到的后半应该就是原样本。
  half = TensorDict(
    {"actor": obs_out["actor"][n:], "critic": obs_out["critic"][n:]},
    batch_size=[n],
  )
  obs_back, _ = mirror_augmentation(env, half, None)
  assert obs_back is not None
  torch.testing.assert_close(obs_back["actor"][n:], obs["actor"])
  torch.testing.assert_close(obs_back["critic"][n:], obs["critic"])


def test_mirror_augmentation_rejects_unknown_obs_term():
  env = _fake_env({"actor": [("mystery_term", (3,))]})
  obs = TensorDict({"actor": torch.randn(2, 3)}, batch_size=[2])
  with pytest.raises(KeyError, match="mystery_term"):
    mirror_augmentation(env, obs, None)


def test_mirror_augmentation_rejects_wrong_action_dim():
  env = _fake_env(_TERMS)
  with pytest.raises(ValueError, match="动作维度"):
    mirror_augmentation(env, None, torch.randn(2, 3))


@pytest.mark.parametrize(
  ("task", "clip", "symmetry"),
  [
    ("Mjlab-Velocity-Flat-Pikachu-V031-T6", None, False),
    ("Mjlab-Velocity-Flat-Pikachu-V031-Clip", 1.0, False),
    ("Mjlab-Velocity-Flat-Pikachu-V031-Sym", None, True),
    ("Mjlab-Velocity-Flat-Pikachu-V031-SymClip", 1.0, True),
    ("Mjlab-Velocity-Flat-Pikachu-V031-YawHuber", None, False),
  ],
)
def test_ablation_arms_set_the_right_fields(task, clip, symmetry):
  """五档消融必须各自生效 —— 这是静默失效的高危点。

  ``clip_actions`` 是 **runner cfg** 的字段（``train.py`` 读
  ``cfg.agent.clip_actions``），不是 algorithm 的。写成
  ``cfg.algorithm.clip_actions`` 不会抛异常，但训练时完全不生效，整档退化成
  基线 —— 只有类型检查/测试才能抓到。
  """
  from mjlab.rl import RslRlOnPolicyRunnerCfg
  from mjlab.tasks.registry import load_rl_cfg

  cfg = load_rl_cfg(task)
  # ``load_rl_cfg`` 的返回类型是基类，``algorithm`` 只在 OnPolicyRunnerCfg 上。
  assert isinstance(cfg, RslRlOnPolicyRunnerCfg)
  assert cfg.clip_actions == clip
  assert (cfg.algorithm.symmetry_cfg is not None) is symmetry
  # 五档的 experiment_name / wandb_project 必须互不相同，否则曲线会混在一起。
  assert cfg.wandb_project


def test_ablation_arms_have_distinct_logging_identity():
  from mjlab.tasks.registry import load_rl_cfg

  tasks = [
    "Mjlab-Velocity-Flat-Pikachu-V031-T6",
    "Mjlab-Velocity-Flat-Pikachu-V031-Clip",
    "Mjlab-Velocity-Flat-Pikachu-V031-Sym",
    "Mjlab-Velocity-Flat-Pikachu-V031-SymClip",
  ]
  names = [load_rl_cfg(t).experiment_name for t in tasks]
  projects = [load_rl_cfg(t).wandb_project for t in tasks]
  assert len(set(names)) == len(tasks)
  assert len(set(projects)) == len(tasks)


def test_yaw_huber_arm_actually_installs_the_reward_term():
  """第 5 档隔离的是**奖励**，不是 runner —— 必须验证环境真的多了那一项。"""
  from mjlab.tasks.registry import load_env_cfg

  t6 = load_env_cfg("Mjlab-Velocity-Flat-Pikachu-V031-T6")
  yh = load_env_cfg("Mjlab-Velocity-Flat-Pikachu-V031-YawHuber")
  assert "angular_velocity_huber" not in t6.rewards
  assert "angular_velocity_huber" in yh.rewards
  term = yh.rewards["angular_velocity_huber"]
  assert term.weight < 0.0
  assert term.params["command_name"] == "twist"
  # delta 应与 ``track_angular_velocity`` 的 std 同量级，否则回复梯度要么
  # 太晚出现（delta 过大）要么把正常步态的本底偏航也罚掉（delta 过小）。
  assert term.params["delta"] == pytest.approx(0.40)
  # 除这一项外，两档的奖励表必须逐项相同 —— 否则就不是单因素对照了。
  assert set(t6.rewards) | {"angular_velocity_huber"} == set(yh.rewards)
