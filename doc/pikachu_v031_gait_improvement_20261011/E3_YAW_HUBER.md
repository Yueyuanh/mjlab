# E3：非饱和 yaw Huber

## 因果问题

E0 的 angular tracking 是指数型，约 1 rad/s 的偏航误差已接近梯度饱和；
body angular penalty 又只罚 roll/pitch。E3 检验补上大误差梯度能否收回直行方向。

## 相对 E2 的唯一改动

新增 `angular_velocity_huber_error`：`delta=0.25 rad/s`，权重 `-0.30`。
指数奖励保留，负责零误差附近精度。筛选预算 5,000 iterations。

## 训练命令

```bash
uv run train Mjlab-Velocity-Flat-Pikachu-V031-E3-YawHuber \
  --env.scene.num-envs 4096 --agent.max-iterations 5000 \
  --agent.save-interval 500 --agent.seed 42 \
  --agent.run-name screen-s42 --agent.logger tensorboard
```

## Play 命令

```bash
uv run play Mjlab-Velocity-Flat-Pikachu-V031-E3-YawHuber \
  --checkpoint-file logs/rsl_rl/pikachu_v031_e3_yaw_huber/<RUN>/model_4999.pt \
  --viewer velocity-joystick
```

## 额外验收

直行 yaw RMS 和每米 heading drift 必须优于 E2；转向档不能因惩罚过强而跟不上
`wz=0.3 rad/s`。


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
`logs/rsl_rl/pikachu_v031_e3_yaw_huber/2026-10-11_21-45-39_screen-s42`。

```bash
uv run train Mjlab-Velocity-Flat-Pikachu-V031-E3-YawHuber \
  --env.scene.num-envs 4096 --agent.max-iterations 1001 \
  --agent.save-interval 500 --agent.seed 42 \
  --agent.run-name screen-s42 --agent.logger tensorboard

uv run play Mjlab-Velocity-Flat-Pikachu-V031-E3-YawHuber \
  --checkpoint-file logs/rsl_rl/pikachu_v031_e3_yaw_huber/2026-10-11_21-45-39_screen-s42/model_1000.pt \
  --viewer velocity-joystick
```

训练末 reward/episode length 为 8.71/214，对照 E2 的 20.42/463 均明显变差。
确定性直行速度仍仅 0.0019 m/s。低 yaw RMS 0.0077 rad/s 是“没有走动”的假改善；
在解决探索方差与裁剪漏洞之前，无法证明 `-0.30` yaw Huber 有效，当前权重偏强。
