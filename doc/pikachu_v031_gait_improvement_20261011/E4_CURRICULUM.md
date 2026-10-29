# E4：分阶段鲁棒训练（最终候选）

## 因果问题

E0 从第 0 轮每 1–3 秒注入满强度速度扰动，且 base linear velocity 噪声
±0.5 m/s 与命令同量级。E4 检验先学直行/对称、再逐步加鲁棒性是否更稳定。

## 相对 E3 的唯一改动

按 policy steps 分三阶段（每 iteration=24 steps）：

1. 0–999 iter：观测噪声 0.2×，push=0；
2. 1000–2999：噪声 0.5×，弱 push（横向 ±0.15 m/s、yaw ±0.20 rad/s，4–8 s）；
3. 3000 起：噪声 1.0×，恢复 E0 满强度 push（1–3 s）。

E4 直接长训 30,000 iterations；m1000/m3000/m5000 同时作为课程转折检查点。

## 训练命令

```bash
uv run train Mjlab-Velocity-Flat-Pikachu-V031-E4-Curriculum \
  --env.scene.num-envs 4096 --agent.max-iterations 30000 \
  --agent.save-interval 500 --agent.seed 42 \
  --agent.run-name final-s42 --agent.logger tensorboard
```

## Play 命令

```bash
uv run play Mjlab-Velocity-Flat-Pikachu-V031-E4-Curriculum \
  --checkpoint-file logs/rsl_rl/pikachu_v031_e4_curriculum/<RUN>/model_29999.pt \
  --viewer velocity-joystick
```

## 额外验收

除 nominal 指标外还要保留 startup DR，并在最终报告中单独标注周期 push 条件；
若 m3000 进入满扰动后 episode length 永久崩塌，则阶段 3 强度过陡。


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
`logs/rsl_rl/pikachu_v031_e4_curriculum/2026-10-11_22-01-49_final-s42`。
本轮先跑到 5k 覆盖所有课程转折；上面的 30k 命令保留为完整设计，但当前结果
不支持直接续训到 30k。

```bash
uv run train Mjlab-Velocity-Flat-Pikachu-V031-E4-Curriculum \
  --env.scene.num-envs 4096 --agent.max-iterations 5001 \
  --agent.save-interval 500 --agent.seed 42 \
  --agent.run-name final-s42 --agent.logger tensorboard
```

关键转折数据：

| iteration | reward | episode length | policy std | 解释 |
|---:|---:|---:|---:|---|
| 1000 | 66.67 | 989.9 | 0.776 | 弱扰动切换点 |
| 2000 | 63.73 | 969.2 | 1.738 | 共享命令范围扩展点 |
| 2500 | 54.47 | 937.7 | 2.295 | 满扰动前最后稳定存档 |
| 3000 | 54.09 | 937.3 | 2.810 | 满扰动切换点 |
| 3100 | 7.11 | 197.3 | 2.931 | 发生断崖 |
| 5000 | 8.80 | 242.0 | 5.818 | 仍未恢复 |

`model_2500.pt` 和 `model_5000.pt` 的确定性前向速度分别为 -0.0013 和
-0.0027 m/s；满 push + 满噪声下分别在 2.78 s 和 2.84 s 摔倒。E4 第三阶段
明确未通过，且训练高回报与确定性部署表现脱节。

精确播放命令：

```bash
# 满扰动前稳定训练检查点（仍是站立策略）
uv run play Mjlab-Velocity-Flat-Pikachu-V031-E4-Curriculum \
  --checkpoint-file logs/rsl_rl/pikachu_v031_e4_curriculum/2026-10-11_22-01-49_final-s42/model_2500.pt \
  --viewer velocity-joystick

# 5k 失败边界样本
uv run play Mjlab-Velocity-Flat-Pikachu-V031-E4-Curriculum \
  --checkpoint-file logs/rsl_rl/pikachu_v031_e4_curriculum/2026-10-11_22-01-49_final-s42/model_5000.pt \
  --viewer velocity-joystick
```
