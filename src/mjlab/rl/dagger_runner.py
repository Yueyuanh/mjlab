"""DAgger + PPO 蒸馏的 runner(PHP 蒸馏 Module 3 的装载侧)。

与 ``MjlabOnPolicyRunner`` 的差别只有三点:

1. 构造前把 ``teacher`` 配置里取值为 ``None`` 的可选项也剥掉(基类只剥
   ``actor`` / ``critic``,教师的 ``MLPModel`` 会因 ``cnn_cfg=None`` 报错)。
2. 构造后从 ``teacher_checkpoint`` 加载冻结教师权重。
3. ONNX 导出走**基类的纯策略导出**:学生是"速度命令 + 本体感受 -> 动作"的
   可部署策略,观测里没有参考动作,不需要像跟踪任务那样把整库参考烘进 ONNX。
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from rsl_rl.env import VecEnv

from mjlab.rl.dagger_ppo import DaggerPPO
from mjlab.rl.runner import MjlabOnPolicyRunner


def _strip_optional_model_cfg(cfg: dict) -> None:
  """剥掉 ``MLPModel`` 不接受的可选构造参数(与 ``MjlabOnPolicyRunner`` 同一套规则)。

  ``cnn_cfg`` / ``distribution_cfg`` 取值为 None 时删掉;``rnn_type`` 为 None 时
  连 ``rnn_hidden_dim`` / ``rnn_num_layers`` 一起删 —— 后两个是 int 不是 None,
  只按"取值为 None"筛会漏掉,直接 ``TypeError``。
  """
  for opt in ("cnn_cfg", "distribution_cfg"):
    if cfg.get(opt) is None:
      cfg.pop(opt, None)
  if cfg.get("rnn_type") is None:
    for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
      cfg.pop(opt, None)


class DaggerOnPolicyRunner(MjlabOnPolicyRunner):
  """学生策略 + 冻结教师的 on-policy runner。

  ``alg`` 实际是 ``DaggerPPO``,但基类把它标成 ``PPO``;这里不重新标注(会被
  pyright 判为不兼容覆盖),用到时 ``cast`` 一次。
  """

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ) -> None:
    # ``actor`` / ``critic`` / ``teacher`` 三网都要剥可选项。
    for key in ("actor", "critic", "teacher"):
      if key in train_cfg:
        _strip_optional_model_cfg(train_cfg[key])
    self.registry_name = registry_name
    super().__init__(env, train_cfg, log_dir, device)

    checkpoint = self.cfg.get("teacher_checkpoint") or ""
    if checkpoint:
      path = Path(checkpoint)
      if not path.exists():
        raise FileNotFoundError(f"teacher checkpoint not found: {path}")
      cast(DaggerPPO, self.alg).load_teacher(str(path))
    else:
      print(
        "[WARN] DaggerOnPolicyRunner: teacher_checkpoint is empty — the teacher "
        "stays randomly initialized, the BC term will teach noise."
      )
