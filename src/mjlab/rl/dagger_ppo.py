"""DAgger + PPO 混合算法(PHP 蒸馏 Module 3 的实现)。

``php/PHP_parkour_distillation_migration.md`` 的核心论点:纯行为克隆
(rsl_rl 自带的 ``Distillation``)只对齐动作、**不看回合结果**,学生一旦漂出
教师分布就没有任何信号把它拉回来;纯 RL 又缺少教师那种"见过整库参考"的先验,
在稀疏奖励下学不动。所以要把两者叠起来::

    L = λ_PPO · L_PPO(策略梯度 + 熵)  +  λ_D · L_BC(对教师动作的模仿)

并且按 PHP 的要求带上四件事:

1. **BC 项全程保留**。λ_D 从 ``lambda_d_start`` 线性退火到 ``lambda_d_final``
   (训练前半段退完),但**永远不为 0** —— 它是 RL 的锚,去掉就退化成纯 PPO。
2. **真终止样本不计 BC**。学生违反专家终止条件(摔倒/偏离参考)时,教师在那
   个状态上的动作没有意义,硬模仿会把学生往错误方向拽。用
   ``dones & ~time_outs`` 精确识别"真终止"(超时不算),把这些样本的 BC 权重
   置 0 —— 它们的纠正信号交给 PPO 的价值函数(负 advantage)。
3. **均匀技能采样**。采样侧不用 adaptive(见 ``velocity_match``),避免蒸馏时
   采样分布朝少数难帧塌缩。
4. **教师冻结**。教师只做前向,不进优化器、不进 ``state_dict`` 的梯度图。

与 ``rsl_rl.algorithms.PPO`` 的关系:继承它复用 ``act`` / ``compute_returns`` /
``process_env_step`` 的骨架,只重写 ``update()``(加 BC 项)、``act()``(顺带
存教师动作)、``process_env_step()``(存 BC mask)与构造/加载(多一个教师网络)。
"""

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict


class DaggerPPO(PPO):
  """PPO 策略梯度 + 教师行为克隆的混合算法。"""

  actor: MLPModel
  critic: MLPModel
  teacher: MLPModel

  def __init__(
    self,
    actor: MLPModel,
    critic: MLPModel,
    teacher: MLPModel,
    storage: RolloutStorage,
    *,
    lambda_d_start: float = 1.0,
    lambda_d_final: float = 0.1,
    lambda_d_anneal_iters: int = 5000,
    bc_loss_type: str = "mse",
    bc_mask_terminations: bool = True,
    **kwargs,
  ) -> None:
    """Initialize the hybrid algorithm.

    Args:
      lambda_d_start: 初始 BC 权重。
      lambda_d_final: 退火终点 BC 权重(>0,BC 全程保留)。
      lambda_d_anneal_iters: 退火持续的迭代数。
      bc_loss_type: ``mse`` 或 ``huber``。
      bc_mask_terminations: 是否把"真终止"样本的 BC 权重置 0。
      **kwargs: 透传给 ``PPO``(clip_param / learning_rate / ... )。
    """
    super().__init__(actor, critic, storage, **kwargs)

    self.teacher = teacher.to(self.device)
    self._raw_teacher = self.teacher
    self.teacher.eval()
    for param in self.teacher.parameters():
      param.requires_grad_(False)

    self.lambda_d_start = lambda_d_start
    self.lambda_d_final = lambda_d_final
    self.lambda_d_anneal_iters = max(int(lambda_d_anneal_iters), 1)
    self.bc_mask_terminations = bc_mask_terminations

    loss_fns = {"mse": nn.functional.mse_loss, "huber": nn.functional.huber_loss}
    if bc_loss_type not in loss_fns:
      raise ValueError(
        f"unknown bc_loss_type {bc_loss_type!r}; expected {list(loss_fns)}"
      )
    self.bc_loss_fn = loss_fns[bc_loss_type]

    # BC 需要的两块额外缓冲。``RolloutStorage("rl", ...)`` 只为 RL 分配
    # values/log_prob/... ,这里补上教师动作与 BC mask(与 actions 同形状)。
    n_steps, n_envs = storage.num_transitions_per_env, storage.num_envs
    storage.privileged_actions = torch.zeros(  # type: ignore[attr-defined]
      n_steps, n_envs, *storage.actions_shape, device=self.device
    )
    storage.bc_mask = torch.ones(  # type: ignore[attr-defined]
      n_steps, n_envs, 1, device=self.device
    )

    # ``update()`` 被调用的次数 == 训练迭代数,用来驱动 λ_D 退火。
    self.num_updates = 0

  # -- 退火 -------------------------------------------------------------------

  @property
  def lambda_d(self) -> float:
    """当前迭代的 BC 权重(线性退火,永不为 0)。"""
    progress = min(self.num_updates / self.lambda_d_anneal_iters, 1.0)
    return self.lambda_d_start + (self.lambda_d_final - self.lambda_d_start) * progress

  # -- rollout ----------------------------------------------------------------

  def act(self, obs: TensorDict) -> torch.Tensor:
    """采样学生动作,同时记录教师在**同一状态**上的动作(DAgger 目标)。"""
    actions = super().act(obs)
    with torch.inference_mode():
      self.transition.privileged_actions = self.teacher(obs).detach()
    return actions

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    """记录一步,并把教师动作与 BC mask 写进额外缓冲。"""
    privileged = self.transition.privileged_actions
    # PHP:学生违反专家终止条件的样本不计 BC。``dones`` 是"终止 or 超时",
    # ``extras["time_outs"]`` 是超时;两者相减剩下的才是真终止(摔倒/偏离)。
    if self.bc_mask_terminations:
      time_outs = extras.get("time_outs")
      if time_outs is None:
        mask = 1.0 - dones.float()
      else:
        mask = 1.0 - dones.float() * (1.0 - time_outs.float())
    else:
      mask = torch.ones_like(dones, dtype=torch.float)

    super().process_env_step(obs, rewards, dones, extras)
    # ``add_transition`` 已把写游标 +1,所以刚写的那格是 ``step - 1``。
    idx = self.storage.step - 1
    self.storage.privileged_actions[idx].copy_(privileged)  # type: ignore[attr-defined]
    self.storage.bc_mask[idx, :, 0].copy_(mask)  # type: ignore[attr-defined]

  # -- 优化 -------------------------------------------------------------------

  def update(self) -> dict[str, float]:
    """跑若干 epoch 的 mini-batch,返回各项 loss 均值。"""
    self.num_updates += 1
    lambda_d = self.lambda_d
    lambda_ppo = 1.0 - lambda_d

    st = self.storage
    batch_size = st.num_envs * st.num_transitions_per_env
    mini_batch_size = batch_size // self.num_mini_batches
    indices = torch.randperm(
      self.num_mini_batches * mini_batch_size, device=self.device
    )

    # ``TensorDict.flatten`` 的类型标注是 ``TensorDict | None``,实际恒非 None。
    observations: TensorDict = st.observations.flatten(0, 1)  # type: ignore[assignment]
    actions = st.actions.flatten(0, 1)
    values = st.values.flatten(0, 1)
    returns = st.returns.flatten(0, 1)
    old_log_prob = st.actions_log_prob.flatten(0, 1)
    advantages = st.advantages.flatten(0, 1)
    assert st.distribution_params is not None, "rollout was never stored"
    old_dist_params = tuple(p.flatten(0, 1) for p in st.distribution_params)
    privileged_actions = st.privileged_actions.flatten(0, 1)  # type: ignore[attr-defined]
    bc_mask = st.bc_mask.flatten(0, 1)  # type: ignore[attr-defined]

    totals = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0, "bc": 0.0}

    for _ in range(self.num_learning_epochs):
      for i in range(self.num_mini_batches):
        start = i * mini_batch_size
        stop = (i + 1) * mini_batch_size
        idx = indices[start:stop]
        batch_obs = observations[idx]

        if self.normalize_advantage_per_mini_batch:
          with torch.no_grad():
            adv = advantages[idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        else:
          adv = advantages[idx]

        # 用当前参数重算 log prob / 熵(PPO 的 importance ratio 需要)。
        self.actor(batch_obs, stochastic_output=True)
        actions_log_prob = self.actor.get_output_log_prob(actions[idx])  # type: ignore[arg-type]
        values_pred = self.critic(batch_obs)
        dist_params = tuple(self.actor.output_distribution_params)  # type: ignore[arg-type]
        entropy = self.actor.output_entropy

        # 自适应学习率(与 PPO 完全一致)。
        if self.desired_kl is not None and self.schedule == "adaptive":
          with torch.inference_mode():
            kl = self.actor.get_kl_divergence(  # type: ignore[arg-type]
              tuple(p[idx] for p in old_dist_params), dist_params
            )
            kl_mean = torch.mean(kl)
            if self.is_multi_gpu:
              torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
              kl_mean /= self.gpu_world_size
            if self.gpu_global_rank == 0:
              if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
              elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)
            if self.is_multi_gpu:
              lr_tensor = torch.tensor(self.learning_rate, device=self.device)
              torch.distributed.broadcast(lr_tensor, src=0)
              self.learning_rate = lr_tensor.item()
            for param_group in self.optimizer.param_groups:
              param_group["lr"] = self.learning_rate

        # PPO 代理目标。
        ratio = torch.exp(actions_log_prob - torch.squeeze(old_log_prob[idx]))
        surrogate = -torch.squeeze(adv) * ratio
        surrogate_clipped = -torch.squeeze(adv) * torch.clamp(
          ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # 价值函数(不乘 λ_PPO:它是 critic 自己的目标,跟着退火一起缩会拖死
        # 价值估计,而 BC 期的策略改进恰恰依赖准确的 V)。
        if self.use_clipped_value_loss:
          value_clipped = values[idx] + (values_pred - values[idx]).clamp(
            -self.clip_param, self.clip_param
          )
          value_loss = torch.max(
            (values_pred - returns[idx]).pow(2),
            (value_clipped - returns[idx]).pow(2),
          ).mean()
        else:
          value_loss = (returns[idx] - values_pred).pow(2).mean()

        # 行为克隆项:学生在自身访问到的状态上对齐教师的**确定性输出**(均值)。
        mask = bc_mask[idx]
        if float(mask.sum()) > 0.0:
          student_mean = self.actor(batch_obs)
          per_sample = self.bc_loss_fn(
            student_mean, privileged_actions[idx], reduction="none"
          ).mean(dim=-1, keepdim=True)
          bc_loss = (per_sample * mask).sum() / mask.sum().clamp(min=1.0)
        else:
          bc_loss = torch.zeros((), device=self.device)

        loss = (
          lambda_ppo * (surrogate_loss - self.entropy_coef * entropy.mean())
          + self.value_loss_coef * value_loss
          + lambda_d * bc_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        if self.is_multi_gpu:
          self.reduce_parameters()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        totals["value"] += value_loss.item()
        totals["surrogate"] += surrogate_loss.item()
        totals["entropy"] += entropy.mean().item()
        totals["bc"] += bc_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    loss_dict = {k: v / num_updates for k, v in totals.items()}
    loss_dict["lambda_d"] = lambda_d
    loss_dict["lambda_ppo"] = lambda_ppo

    self.storage.clear()
    return loss_dict

  # -- 模式 -------------------------------------------------------------------

  def train_mode(self) -> None:
    """学生/评论家训练,教师永远 eval。"""
    self.actor.train()
    self.critic.train()
    self.teacher.eval()

  def eval_mode(self) -> None:
    self.actor.eval()
    self.critic.eval()
    self.teacher.eval()

  # -- 存取 -------------------------------------------------------------------

  def save(self) -> dict:
    saved = super().save()
    saved["teacher_state_dict"] = self._raw_teacher.state_dict()
    return saved

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    """加载。``actor_state_dict`` 视为**教师**权重(与 rsl_rl Distillation 同约定)。"""
    if load_cfg is None:
      load_cfg = {
        "actor": "actor_state_dict" in loaded_dict,
        "critic": "critic_state_dict" in loaded_dict,
        "teacher": True,
        "optimizer": "optimizer_state_dict" in loaded_dict,
        "iteration": "actor_state_dict" not in loaded_dict,
      }
    if load_cfg.get("teacher"):
      state = loaded_dict.get("teacher_state_dict") or loaded_dict.get(
        "actor_state_dict"
      )
      if state is None:
        raise KeyError(
          "no teacher weights in checkpoint (need 'teacher_state_dict' or "
          "'actor_state_dict')"
        )
      self._raw_teacher.load_state_dict(state, strict=strict)
    return super().load(loaded_dict, load_cfg, strict)

  def load_teacher(self, path: str, strict: bool = True) -> None:
    """从教师 checkpoint(跟踪任务训练的 ``model_*.pt``)加载教师网络。"""
    ckpt = torch.load(path, map_location=self.device, weights_only=False)
    state = ckpt.get("teacher_state_dict") or ckpt.get("actor_state_dict")
    if state is None:
      raise KeyError(
        f"{path}: no 'actor_state_dict'/'teacher_state_dict' — "
        f"got keys {sorted(ckpt.keys())}"
      )
    self._raw_teacher.load_state_dict(state, strict=strict)
    self.teacher.eval()
    for param in self.teacher.parameters():
      param.requires_grad_(False)
    print(f"[INFO] DaggerPPO: teacher loaded from {path}")

  def get_policy(self) -> MLPModel:
    return self._raw_actor

  @staticmethod
  def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str):
    """构造 actor / critic / **teacher** 三网 + RL 存储 + 本算法。"""
    alg_class: type[DaggerPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore[assignment]
    actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore[assignment]
    critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore[assignment]
    teacher_class: type[MLPModel] = resolve_callable(cfg["teacher"].pop("class_name"))  # type: ignore[assignment]

    # 与 ``PPO.construct_algorithm`` 同一套解析:默认观测集多一个 ``teacher``。
    default_sets = ["actor", "critic", "teacher"]
    if cfg["algorithm"].get("rnd_cfg") is not None:
      default_sets.append("rnd_state")
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

    cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
    cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

    actor: MLPModel = actor_class(
      obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]
    ).to(device)
    print(f"Student Model: {actor}")
    if cfg["algorithm"].pop("share_cnn_encoders", None):
      cfg["critic"]["cnns"] = actor.cnns  # type: ignore[attr-defined]
    critic: MLPModel = critic_class(
      obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]
    ).to(device)
    print(f"Critic Model: {critic}")
    teacher: MLPModel = teacher_class(
      obs, cfg["obs_groups"], "teacher", env.num_actions, **cfg["teacher"]
    ).to(device)
    print(f"Teacher Model: {teacher}")

    storage = RolloutStorage(
      "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
    )
    alg: DaggerPPO = alg_class(
      actor,
      critic,
      teacher,
      storage,
      device=device,
      **cfg["algorithm"],
      multi_gpu_cfg=cfg["multi_gpu"],
    )
    alg.compile(cfg.get("torch_compile_mode"))
    return alg

  def broadcast_parameters(self) -> None:
    """多卡同步:学生 + 评论家 + 教师。"""
    params = [
      self._raw_actor.state_dict(),
      self._raw_critic.state_dict(),
      self._raw_teacher.state_dict(),
    ]
    torch.distributed.broadcast_object_list(params, src=0)
    self._raw_actor.load_state_dict(params[0])
    self._raw_critic.load_state_dict(params[1])
    self._raw_teacher.load_state_dict(params[2])

  def reduce_parameters(self) -> None:
    """只对**学生**做梯度平均(教师冻结、评论家无多卡需求差异)。"""
    all_params = list(chain(self.actor.parameters(), self.critic.parameters()))
    grads = [p.grad.view(-1) for p in all_params if p.grad is not None]
    all_grads = torch.cat(grads)
    torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
    all_grads /= self.gpu_world_size
    offset = 0
    for param in all_params:
      if param.grad is not None:
        numel = param.numel()
        param.grad.data.copy_(
          all_grads[offset : offset + numel].view_as(param.grad.data)
        )
        offset += numel


__all__ = ["DaggerPPO"]
