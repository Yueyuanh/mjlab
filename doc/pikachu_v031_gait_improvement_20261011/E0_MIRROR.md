# E0-Mirror：E0 上的低权重左右镜像实验

## 目标

只验证低权重镜像约束能否修复 E0 的左右 hip-roll 不对称，同时保留 E0
已经学到的行走能力。首选系数为 `0.05`；`0.10` 是上限对照，不应与首选任务
同时启动。

## 相对 E0 的改动

- actor/critic observation 与 action 做左右镜像数据增强；
- policy mirror loss 开启，首选系数 `0.05`，上限系数 `0.10`；
- critic 的世界系足端接触力转为 base-heading frame，这是正确镜像该观测的
  必要坐标变换，不改变 actor 输入、奖励或动力学；
- 其余保持 E0：无 raw action clip、无 processed action clip、无 yaw Huber、
  无新增 curriculum，PPO/奖励/扰动/噪声均不变。

## 注册任务

- 首选：`Mjlab-Velocity-Flat-Pikachu-V031-E0-Mirror005`
- 上限对照：`Mjlab-Velocity-Flat-Pikachu-V031-E0-Mirror010`

两个任务默认都是 50,000 iterations、每 500 iterations 保存。每个任务使用
独立的 `experiment_name`、W&B project 和日志目录。

## 首选 0.05：训练命令

```bash
MUJOCO_GL=egl uv run train \
  Mjlab-Velocity-Flat-Pikachu-V031-E0-Mirror005 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 50000 \
  --agent.save-interval 500 \
  --agent.seed 42 \
  --agent.run-name mirror005-s42 \
  --agent.logger tensorboard \
  --video True \
  --video-length 200
```

不要显式传 `--video-interval`。训练器会自动取
`save_interval × num_steps_per_env = 500 × 24 = 12000` policy steps，因此每个
500-iteration 保存点都会触发一段 200 帧视频。

产物目录：

```text
logs/rsl_rl/pikachu_v031_e0_mirror005/<RUN>/
├── model_0.pt, model_500.pt, ..., model_49500.pt, model_49999.pt
├── <RUN>.onnx
├── params/env.yaml
├── params/agent.yaml
└── videos/train/rl-video-step-*.mp4
```

ONNX 是当前 checkpoint 的滚动导出文件，后一次保存会覆盖前一次；历史训练状态
由各个 `model_*.pt` 完整保留。

## 首选任务 Play 命令

```bash
uv run play Mjlab-Velocity-Flat-Pikachu-V031-E0-Mirror005 \
  --checkpoint-file \
  logs/rsl_rl/pikachu_v031_e0_mirror005/<RUN>/model_49999.pt \
  --viewer velocity-joystick
```

## 0.10 上限对照

只有 0.05 能稳定行走但 hip-roll 左右绝对均值差仍大于 `0.03 rad` 时，才启动：

```bash
MUJOCO_GL=egl uv run train \
  Mjlab-Velocity-Flat-Pikachu-V031-E0-Mirror010 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 50000 \
  --agent.save-interval 500 \
  --agent.seed 42 \
  --agent.run-name mirror010-s42 \
  --agent.logger tensorboard \
  --video True \
  --video-length 200
```

## 早期检查

虽然完整预算是 50,000，仍应在 `model_500.pt` 和对应视频生成后立即检查。只有
确定性策略确实前进，而不是通过站立获得表面对称，才继续长训。建议门槛：

- 直行 `vx >= 0.30 m/s`；
- 左右 hip-roll 绝对均值差 `< 0.03 rad`；
- yaw RMS `< 0.25 rad/s`。
