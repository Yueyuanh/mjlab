# Pikachu V031 B1 天然有界动作实验设计

## 目的

验证 E0-Mirror005 在不依赖高斯动作硬截断的情况下，能否同时保留行走能力、
左右 `hip_roll` 对称性，并把策略输出和关节位置目标保持在部署安全包络内。

本实验只改变动作分布和边界，不改变当前 XML、电机分组、奖励函数或 PD。这样若
B1 不能行走，可以把原因优先定位到动作空间/探索方式，而不会与 PD 调参混杂。

## B1 配置

| 项目 | B1-BetaBounded |
|---|---|
| 基线 | E0-Mirror005，`mirror_loss_coeff=0.05` |
| 策略分布 | `BetaDistribution`，天然映射到 `[-1, 1]` |
| 分布约束 | `alpha = softplus(x)+1`、`beta = softplus(x)+1`，单峰 |
| runner 保护 | `clip_actions=1.0`，正常情况下不应触发 |
| 目标保护 | hip pitch ±0.50、roll/yaw ±0.20、knee ±0.40、ankle ±0.35 rad |
| PPO entropy | 0.01，与 E0 保持一致以隔离分布变化 |
| PD | 保持当前 Kp/Kd，不在 B1 中调参 |
| 训练 | 4096 env、50,000 iterations、seed 42 |
| 存档 | 每 500 iterations 保存 checkpoint 和训练视频 |

任务名：`Mjlab-Velocity-Flat-Pikachu-V031-B1-BetaBounded`

代码版本：`4937a13c15261321392bc1b9ca0d055d6d90d71a`

训练时资产快照：

- `pikachu_v031_train.xml` SHA-256：
  `d68be1ade235303ccc2230dedbfa286051834d1d6728e3453eddf4fd41507866`
- `pikachu_v031.urdf` SHA-256：
  `6bfd5f9984873e623f54b971d8a43d351404fda25d8f90f89070f351e6aa9b1c`

## 电机与 PD 口径

按当前 V031 配置，hip pitch/roll/yaw 和 knee_up 均按 J4310：12.5 N·m、
12.57 rad/s；ankle 按 J3507：3.0 N·m、15.71 rad/s。B1 不改这些值。

当前 PD 为：

| 分组 | Kp (N·m/rad) | Kd (N·m·s/rad) |
|---|---:|---:|
| J4310 hip/knee | 8.70894 | 0.55443 |
| J3507 ankle | 11.65022 | 0.74168 |

无接触阶跃/扫频共 70 个 case，全部扭矩饱和率为 0。测得左右一致的近似带宽：
hip pitch 2.42 Hz、hip roll 2.19 Hz、hip yaw 1.83 Hz、ankle 2.01 Hz、
knee 1.19 Hz。ankle 阶跃峰值约 2.52 N·m，是正式行走评估中的重点监控项。

`response` 工具的电机名称和速度字段存在旧映射：它会把部分 hip/knee 标为 J3507，
并显示 joint 元数据中的 10/20 rad/s。该标签不作为本实验电机口径；训练与验收均以
上述 V031 actuator group 的 12.5/3.0 N·m 和 12.57/15.71 rad/s 为准。

## 分阶段验收

### m500 / m1000 运动门槛

固定 0.4 m/s、nominal、确定性策略、seed 42、8 秒：

- 实际 `vx >= 0.30 m/s`；
- 8 秒不中止；
- 左右 hip_roll 绝对均值差 `< 0.03 rad`；
- runner clip 触发率必须为 0；
- 处理后目标越本实验包络必须为 0；
- yaw RMS 先用 `< 1.2 rad/s` 作为“能运动”筛查，不把它冒充最终验收。

若 m500 未通过但 m1000 通过，可以继续；若 m1000 仍近似站桩，则停止 B1，分析
Beta 初始探索、动作变化率惩罚和目标幅度，不直接增大 PD。

### 最终验收

- 六命令：stand、fwd、fwd_fast、back、lat、turn；
- nominal、startup DR、完整训练条件；
- 直行 `vx >= 0.30 m/s`；
- hip_roll 绝对均值差 `< 0.03 rad`，p2p 比 0.8–1.25；
- 零 yaw 命令下 yaw RMS `< 0.25 rad/s`、8 秒 heading drift `< 10 deg`；
- `turn +0.3 rad/s` 有同向明显响应；
- 原始动作严格在 `[-1,1]`，runner clip 触发率 0；
- 目标越配置包络率 0；
- 负载行走时 J4310/J3507 扭矩和速度均不越规格，并重点检查 ankle 余量。

只有行走时出现明显目标跟踪滞后、同时策略并未长期贴 Beta 边界，才设计 B2-PD；
否则保持 PD，把后续改动放在 yaw 奖励或动作时序上。

## 启动与回放命令

正式训练：

```bash
cd /home/finnox-4090/RL/Pikachu/mjlab
MUJOCO_GL=egl uv run train \
  Mjlab-Velocity-Flat-Pikachu-V031-B1-BetaBounded \
  --env.scene.num-envs 4096 \
  --env.seed 42 \
  --agent.max-iterations 50000 \
  --agent.save-interval 500 \
  --agent.run-name beta-bounded-mirror005-s42 \
  --video True \
  --video-length 200
```

交互回放：

```bash
cd /home/finnox-4090/RL/Pikachu/mjlab
uv run play Mjlab-Velocity-Flat-Pikachu-V031-B1-BetaBounded \
  --agent.resume True \
  --agent.load-run <RUN_DIR> \
  --agent.load-checkpoint model_49999.pt
```

固定命令指标：

```bash
cd /home/finnox-4090/RL/Pikachu/mjlab
uv run python scripts/record_v03_metrics.py \
  --task Mjlab-Velocity-Flat-Pikachu-V031-B1-BetaBounded \
  --checkpoint <RUN_DIR>/model_49999.pt \
  --out doc/pikachu_v031_beta_bounded_20261013/metrics/final_nominal \
  --tag b1-m49999-nominal \
  --clips stand,fwd,fwd_fast,back,lat,turn \
  --seconds 8 --device cpu --seed 42 --nominal
```

离屏视频：

```bash
cd /home/finnox-4090/RL/Pikachu/mjlab
MUJOCO_GL=egl uv run python scripts/record_v03_video.py \
  --task Mjlab-Velocity-Flat-Pikachu-V031-B1-BetaBounded \
  --checkpoint <RUN_DIR>/model_49999.pt \
  --out doc/pikachu_v031_beta_bounded_20261013/videos \
  --tag b1-m49999-nominal \
  --clips fwd,turn --views side,front \
  --seconds 6 --device cuda:0 --seed 42 --nominal
```

PD 响应复现：

```bash
cd /home/finnox-4090/RL/Pikachu/mjlab
uv run response Mjlab-Velocity-Flat-Pikachu-V031-B1-BetaBounded \
  --mode all --device cpu \
  --output-root doc/pikachu_v031_beta_bounded_20261013/pd_response
```
