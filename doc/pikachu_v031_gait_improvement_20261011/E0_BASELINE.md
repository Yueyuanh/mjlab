# E0：原 T6 基线（冻结对照）

## 目的与唯一角色

不引入改动，复用完整 30k 旧训练作为对照。它回答“新方案比哪一个真实模型好”，
也验证旧模型的偏航与左右 hip-roll DC 偏置可以在统一回放中重现。

## 自变量

无：`clip_actions=None`、processed target 无 clip、无 symmetry、无 yaw Huber、
从第 0 轮使用原始满强度 push/观测噪声。

## 已有运行

`logs/rsl_rl/pikachu_v031_velocity/2026-10-10_20-11-53`，seed 42，4096 env，
30,000 iterations。最终 checkpoint SHA-256：
`3295b9b14dfad53d386eb565aea01e8f51ac7e8a06e6b997ccc308bc86360a1b`。

## 从零复现命令

```bash
uv run train Mjlab-Velocity-Flat-Pikachu-V031-T6 \
  --env.scene.num-envs 4096 --agent.max-iterations 30000 \
  --agent.save-interval 500 --agent.seed 42
```

## Play 命令

```bash
uv run play Mjlab-Velocity-Flat-Pikachu-V031-T6 \
  --checkpoint-file logs/rsl_rl/pikachu_v031_velocity/2026-10-10_20-11-53/model_29999.pt \
  --viewer velocity-joystick
```


## 固定训练条件

- 代码分支：`experiment/v031-gait-symmetry-20261011`
- 冻结提交：`ddcffc9`
- GPU / 环境数：RTX 4090 / 4096
- seed：42（本轮因果筛选）；最终结论需要后续多 seed 复核
- PPO：24 steps/env，5 epochs，4 minibatches，网络 512-256-128
- 保存间隔：500 iterations
- 日志：TensorBoard；每个任务使用独立 `experiment_name`

## 统一评测

每个候选的最后 checkpoint 用同一脚本、同一 seed、同一固定命令评测：

```bash
MUJOCO_GL=egl uv run python scripts/record_v03_metrics.py \
  --task <TASK> --checkpoint <CHECKPOINT> \
  --out doc/pikachu_v031_gait_improvement_20261011/metrics/<EXP> \
  --tag <EXP> --clips stand,fwd,fwd_fast,back,lat,turn \
  --seconds 8 --seed 42 --nominal
```

验收主指标：8 秒直行 heading drift <10°、yaw RMS <0.25 rad/s、左右
hip-roll DC 幅值差 <0.03 rad、p2p 比 0.8–1.25、处理后目标越硬限位 0%。

## 本轮统一回放结果

旧运行与最终权重均保持不变，并通过 SHA-256 校验。`model_1000.pt` 在 0.4 m/s
直行中实际速度 0.399 m/s、yaw RMS 1.131 rad/s、8 秒航向漂移 +91.4°；
`model_29999.pt` 实际速度 0.421 m/s、yaw RMS 0.883 rad/s、漂移 -76.7°。
后者左右 hip-roll 绝对均值差 0.0921 rad，目标越 ±0.26 rad 的比例 26.7%。
这重现了原报告中的“能跟速度，但严重侧偏且长期贴限位”。

等预算检查点播放：

```bash
uv run play Mjlab-Velocity-Flat-Pikachu-V031-T6 \
  --checkpoint-file logs/rsl_rl/pikachu_v031_velocity/2026-10-10_20-11-53/model_1000.pt \
  --viewer velocity-joystick
```
