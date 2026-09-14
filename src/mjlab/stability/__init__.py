"""基于 ``pikachu_27dof_constants`` 的裸 MuJoCo sim2sim / 稳定性验算模块。

旁路 mjlab 高层 env（warp / manager / rsl_rl），在 CPU Mujoco 上直接组装
27-dof 训练体 -- 纯模型导入、策略加载、观测对齐与 PD 控制。每个关注点一个文件：

  ``sim2sim_model.py``  模型导入（地面 + position 执行器 + 起步姿态）
  （后续）``policy.py``  ONNX / JIT 策略加载
  （后续）``obs.py``     观测对齐（velocity / tracking）
  （后续）``main.py``    v025 式主循环与命令行入口
"""

from mjlab.stability.sim2sim_model import (
  KNEES_BENT_START,
  apply_keyframe,
  build_model,
)

__all__ = ["KNEES_BENT_START", "apply_keyframe", "build_model"]
