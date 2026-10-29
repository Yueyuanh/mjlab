# E1：双层动作安全边界

## 因果问题

旧模型的 hip-roll 目标约 30% 超过 ±0.26 rad 硬限位；对旧权重临时加 raw clip
会在 2.8 秒摔倒。E1 从头训练，判断饱和控制是不是偏置形成的必要条件。

## 相对 E0 的唯一改动

- runner `clip_actions=1.0`；
- `JointPositionActionCfg.clip` 冗余限制处理后目标：hip roll/yaw ±0.20、hip pitch
  ±0.50、knee_up ±0.40、ankle ±0.35 rad。

奖励、扰动、噪声均保持 E0。筛选预算 5,000 iterations。

## 训练命令

```bash
uv run train Mjlab-Velocity-Flat-Pikachu-V031-E1-Clip \
  --env.scene.num-envs 4096 --agent.max-iterations 5000 \
  --agent.save-interval 500 --agent.seed 42 \
  --agent.run-name screen-s42 --agent.logger tensorboard
```

## Play 命令

```bash
uv run play Mjlab-Velocity-Flat-Pikachu-V031-E1-Clip \
  --checkpoint-file logs/rsl_rl/pikachu_v031_e1_clip/<RUN>/model_4999.pt \
  --viewer velocity-joystick
```

## 判停规则

若 3k–5k 时 episode length 仍不增长或 8 秒直行无法存活，说明当前动作尺度下
控制权不足；不能通过放开 hip-roll 限位“修好”回报。


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
`logs/rsl_rl/pikachu_v031_e1_clip/2026-10-11_21-15-42_screen-s42`。
原 5k 命令启动后在 `model_1000.pt` 安全落盘后停止；复现所选检查点的定长命令：

```bash
uv run train Mjlab-Velocity-Flat-Pikachu-V031-E1-Clip \
  --env.scene.num-envs 4096 --agent.max-iterations 1001 \
  --agent.save-interval 500 --agent.seed 42 \
  --agent.run-name screen-s42 --agent.logger tensorboard
```

精确播放命令：

```bash
uv run play Mjlab-Velocity-Flat-Pikachu-V031-E1-Clip \
  --checkpoint-file logs/rsl_rl/pikachu_v031_e1_clip/2026-10-11_21-15-42_screen-s42/model_1000.pt \
  --viewer velocity-joystick
```

8 秒标称直行时目标越限为 0%，但实际前向速度只有 0.0023 m/s；确定性 raw action
最大值 11.26，68.8% 元素被 runner 截断。结论：安全边界有效，但“硬裁剪 +
未约束 Gaussian std + entropy=0.01”形成饱和/高熵漏洞，E1 未通过步态验收。
