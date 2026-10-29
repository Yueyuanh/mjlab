# E2：V031 专用左右镜像学习

## 因果问题

在 E1 安全边界上加入对称先验，判断左右 hip-roll DC 偏置是否源自策略优化的
自发对称破缺，而不是模型参数差异。

## 相对 E1 的唯一学习改动

- RSL-RL data augmentation 开启；
- policy mirror loss 开启，系数 0.30；
- V031 实测符号：hip pitch/roll/yaw、ankle 反号，knee_up 同号；
- actor、critic、action 全部镜像；net contact force 先转入 base-heading frame。

critic 足端力只改变坐标表达，不改变维度。筛选预算 5,000 iterations。

## 训练命令

```bash
uv run train Mjlab-Velocity-Flat-Pikachu-V031-E2-Symmetry \
  --env.scene.num-envs 4096 --agent.max-iterations 5000 \
  --agent.save-interval 500 --agent.seed 42 \
  --agent.run-name screen-s42 --agent.logger tensorboard
```

## Play 命令

```bash
uv run play Mjlab-Velocity-Flat-Pikachu-V031-E2-Symmetry \
  --checkpoint-file logs/rsl_rl/pikachu_v031_e2_symmetry/<RUN>/model_4999.pt \
  --viewer velocity-joystick
```

## 额外验收

`Loss/symmetry` 必须有限并总体下降；左右 hip-roll DC 幅值差应明显小于 E1，
同时不能以“两个腿都不动”换取表面对称。


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

## 本轮执行结果

实际运行：
`logs/rsl_rl/pikachu_v031_e2_symmetry/2026-10-11_21-29-49_screen-s42`。

```bash
uv run train Mjlab-Velocity-Flat-Pikachu-V031-E2-Symmetry \
  --env.scene.num-envs 4096 --agent.max-iterations 1001 \
  --agent.save-interval 500 --agent.seed 42 \
  --agent.run-name screen-s42 --agent.logger tensorboard

uv run play Mjlab-Velocity-Flat-Pikachu-V031-E2-Symmetry \
  --checkpoint-file logs/rsl_rl/pikachu_v031_e2_symmetry/2026-10-11_21-29-49_screen-s42/model_1000.pt \
  --viewer velocity-joystick
```

`Loss/symmetry=0.00208` 且有限。标称直行 hip-roll 绝对均值差从 E1 的
0.1042 rad 降到 0.0268 rad（-74.3%），目标越限仍为 0%；但实际前向速度同样
只有 0.0022 m/s，65.1% raw action 被裁剪。结论：镜像映射本身有效，但结果是
“对称站立”而非对称行走，不能作为最终成功证据。
