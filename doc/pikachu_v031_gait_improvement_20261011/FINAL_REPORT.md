# Pikachu V031 步态偏置改进实验：最终报告

日期：2026-10-11

分支：experiment/v031-gait-symmetry-20261011

训练实现提交：ddcffc9

seed / GPU / 并行环境：42 / RTX 4090 / 4096

## 结论

本轮**没有得到可部署的新行走模型**，但把问题进一步定位清楚了：

1. 原模型 E0 确实能跟随 0.4 m/s 命令，但 8 秒直行偏航 -76.7°、yaw RMS
   0.883 rad/s、hip-roll 左右绝对均值差 0.0921 rad，26.7% hip-roll 目标超过
   ±0.26 rad 硬限位。
2. E1 的动作/关节目标裁剪把越限降到 0%；E2 的镜像学习又把等预算 hip-roll
   均值差从 0.1042 降到 0.0268 rad。安全边界和镜像映射本身有效。
3. 但 E1–E4 的确定性策略没有学会走：0.4 m/s 命令下实际速度只有
   -0.003–0.002 m/s。它们较小的 yaw 和左右差主要来自“站住”，不能算改进。
4. 首要根因是**训练配置**：硬动作裁剪与未约束 Gaussian 方差、entropy=0.01
   组合后产生高熵漏洞。E4 policy std 从 0.776 增至 5.818，确定性 raw action
   最大值从 5.59 增至 44.98，约 60%–69% 动作被截断；训练回报与部署时 actor
   mean 严重脱节。
5. E4 在 m3000 把推扰从弱档直接切到原始满强度后，episode length 从 937
   跌至 197，m5000 仍只有 242。第三阶段也确实过陡。
6. E3 的 yaw Huber 不是当前主解：在上述配置混淆下，它把 m1000 reward /
   episode length 从 E2 的 20.42/463 降到 8.71/214，无法证明收益。

因此，对“模型配置还是奖励函数”的回答是：**当前失败首先是策略分布、裁剪和
扰动课程配置问题；奖励函数是次要问题，应在确定性策略真正走起来之后再调。**

## 实验范围与公平口径

原设计是 E1–E3 各 5k 筛选、E4 30k 长训。为在本轮先完成因果筛选，实际执行：

| 实验 | 实际训练 | 统一比较检查点 | 角色 |
|---|---:|---:|---|
| E0 | 既有 30k | m1000、m29999 | 原始基线 |
| E1 | 到 m1046 | m1000 | 仅动作安全边界 |
| E2 | 到 m1000 | m1000 | E1 + 对称学习 |
| E3 | 到 m1000 | m1000 | E2 + yaw Huber |
| E4 | 到 m5000 | m1000/m2500/m5000 | 分阶段噪声与推扰 |

E0–E4 的因果表统一使用 m1000；E0/m29999 与 E4/m2500、m5000 只用于现实模型
比较，不冒充等预算结论。所有回放均为 deterministic actor、seed 42、标称机构、
固定命令、8 秒；另对 E4 加测 startup DR + 满噪声 + 满推扰。当前是单 seed
筛选，不是统计显著性或上机放行结论。

完整运行、事件文件、检查点路径与 SHA-256 见
[run_results.json](manifests/run_results.json)；原始设计见
[experiment_plan.json](manifests/experiment_plan.json)。

## 数据结果

### 等预算 m1000，0.4 m/s 标称直行

| 实验 | 实际 vx m/s | yaw RMS rad/s | 航向变化 ° | hip-roll 均值差 rad | 目标越限 | raw clip 比例 |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 0.3987 | 1.1307 | +91.40 | 0.0835 | 28.27% | 不适用¹ |
| E1 | 0.0023 | 0.0317 | +2.32 | 0.1042 | 0% | 68.85% |
| E2 | 0.0022 | 0.0057 | -8.81 | 0.0268 | 0% | 65.13% |
| E3 | 0.0019 | 0.0077 | -3.56 | 0.0606 | 0% | 69.30% |
| E4 | -0.0011 | 0.0003 | +0.01 | 0.0021 | 0% | 60.00% |

¹ E0 没有 runner clip；其遥测里的“超过单位动作”比例不代表实际裁剪率，故记为
不适用。

E2 相对 E1 的 hip-roll 均值差下降 74.3%，E4 相对 E1 下降 98.0%，但它们
没有产生前向位移。这说明镜像约束确实作用到了姿态，却没有解决速度策略。

![m1000 回放比较](plots/rollout_comparison.png)

### 旧最终模型与 E4 检查点

| 检查点 | 实际 vx m/s | yaw RMS | 航向变化 | roll 均值差 | 目标越限 | raw action max | clip 比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 m29999 | 0.4215 | 0.8830 | -76.66° | 0.0921 | 26.67% | 3.08 | 不适用 |
| E4 m2500 | -0.0013 | 0.0015 | +1.72° | 0.0225 | 0% | 17.46 | 60.00% |
| E4 m5000 | -0.0027 | 0.0145 | +6.91° | 0.1018 | 0% | 44.98 | 69.17% |

E4 的 yaw 数字更小是因为没有走；真正关键的速度指标下降了约 100%。m5000
还把 m2500 的对称性破坏掉，不能作为候选模型。

![最终检查点比较](plots/final_candidate_comparison.png)

![hip-roll 镜像时序](plots/hip_roll_timeseries.png)

### E4 课程转折

| iteration | reward | episode length | std | 事件 |
|---:|---:|---:|---:|---|
| 1000 | 66.67 | 989.9 | 0.776 | 0.2×噪声/零推扰 → 0.5×/弱推扰 |
| 2000 | 63.73 | 969.2 | 1.738 | 共享速度命令范围扩展 |
| 2500 | 54.47 | 937.7 | 2.295 | 满推扰前存档 |
| 3000 | 54.09 | 937.3 | 2.810 | 切入满噪声/满推扰 |
| 3100 | 7.11 | 197.3 | 2.931 | 断崖 |
| 4000 | 8.06 | 224.8 | 4.236 | 未恢复 |
| 5000 | 8.80 | 242.0 | 5.818 | 未恢复 |

原 E0 的 std 在 m1000/m2500/m29999 分别只有 0.305/0.329/0.284。E4 的增长
不是正常的探索退火，而是持续发散。

![训练曲线](plots/training_comparison.png)

### 满扰动复核

E4/m2500 在 startup DR + 满观测噪声 + 满推扰下直行 2.78 s 摔倒；E4/m5000
为 2.84 s。六个命令档的 m2500 存活时间为 2.74–2.86 s，m5000 为
2.82–3.48 s。第三阶段没有获得合格鲁棒性。

完整 66 组结果：

- [analysis_summary.json](analysis_summary.json)
- [rollout_summary.csv](rollout_summary.csv)
- [逐步遥测目录](metrics/)

## 视频

六段视频均已用 ffprobe 验证为 H.264、1280×720、50 fps、8.0 s：

| 检查点 | 侧视 | 正视 |
|---|---|---|
| E0 m29999 | [MP4](videos/fwd__side__E0_model29999_nominal.mp4) | [MP4](videos/fwd__front__E0_model29999_nominal.mp4) |
| E4 m2500 | [MP4](videos/fwd__side__E4_model2500_nominal.mp4) | [MP4](videos/fwd__front__E4_model2500_nominal.mp4) |
| E4 m5000 | [MP4](videos/fwd__side__E4_model5000_nominal.mp4) | [MP4](videos/fwd__front__E4_model5000_nominal.mp4) |

## 分实验判定

- **E1：安全边界有效，步态失败。** 越限 28.3%→0%，但 raw action 大量饱和，
  确定性策略站立。不能继续只靠加大动作范围解决。
- **E2：对称先验有效，仍非步态。** symmetry loss 有限，左右差明显下降；
  结论只支持保留镜像实现，不支持当前权重部署。
- **E3：本轮不支持。** yaw Huber 权重 -0.30 在 m1000 明显降低训练表现，
  且静止策略让 yaw 指标失去解释力。
- **E4：前两阶段训练曲线稳定，第三阶段失败。** m2500 只是训练期“稳定”而不是
  部署候选；m5000 是用于确认课程断崖的失败样本。

## 根因分析

### 1. 硬裁剪与 entropy 的目标不一致

GaussianDistribution 默认 std 上限为 1e6；当前 entropy coefficient 为 0.01。
PPO 对裁剪前 Gaussian 计算 log-prob/entropy，环境却只执行裁剪后的 ±1 动作。
当 std 或 mean 超出边界后，继续增大方差仍能获得 entropy 收益，而环境看到的
动作几乎不变。这解释了：

- E1–E4 的 std 持续增大；
- 60%–69% 动作被裁剪；
- 确定性 actor mean 输出 11–45 的异常幅值；
- 训练中随机策略有回报，play 时 mean 策略只会站立。

这是本轮新方案最主要的配置错误。

### 2. 满推扰跨度过大

E4 从横向 ±0.15 m/s、yaw ±0.20 rad/s、4–8 s，直接跳到平移最高
±0.5 m/s、姿态 ±0.52/0.78 rad/s、1–3 s。m3000 后的即时断崖及 2k 轮不恢复
说明该跨度不适合作为单级课程。

### 3. 奖励调整被上游问题遮蔽

E3 的 Huber 提供了大 yaw 误差梯度，但策略没有行走时，yaw 误差可以通过静止
轻易变小。应先修复动作分布和确定性速度跟踪，再以较小权重复测奖励。

## 下一轮建议

按以下顺序做新的单因素链，不建议直接续训当前 E4：

1. **E5：先封住探索漏洞。** 从 E1 开始，把 actor distribution 改为
   log-std、初始 std 约 0.35、std_range 建议 (0.05, 0.6)，entropy coefficient
   从 0.01 降到 0.001 或 0；保留 processed joint target clip。更彻底的长期方案
   是使用带正确 log-prob 修正的 tanh-squashed Gaussian，而不是事后硬裁剪。
2. **E6：重新验证对称项。** E5 成功行走后再加 E2 镜像；要求速度不下降，同时
   hip-roll 均值差 <0.03 rad。
3. **E7：小权重 yaw Huber。** 从 -0.05 或 -0.10 起，不直接使用 -0.30；同时
   检查转向档 wz=0.3 的跟踪，防止“只会直行/站立”。
4. **E8：细化扰动课程。** 至少拆成 zero → weak → medium → strong 四级；把
   interval 从 6–10 s、4–8 s、3–6 s 逐步缩短。若实机不会遇到竖直/roll/pitch
   瞬时速度注入，不应把这些最大项照搬为训练主扰动。
5. **增加在线闸门。** 每 500 iterations 自动跑 deterministic 8 秒回放；要求
   fwd vx ≥0.30 m/s、raw clip fraction ≤10%、目标越限 0%、yaw RMS <0.25、
   hip-roll 均值差 <0.03。未过速度闸门时，任何低 yaw/高对称分都不计成功。
6. 单 seed 通过后再跑 seeds 42/43/44；最后候选才进入 30k 和上机测试。

## 启动与 Play 命令

每个实验的完整设计命令与判停规则在
[E0](E0_BASELINE.md)、[E1](E1_ACTION_CLIP.md)、[E2](E2_SYMMETRY.md)、
[E3](E3_YAW_HUBER.md)、[E4](E4_CURRICULUM.md)。本轮实际命令如下：

```bash
# E0 从零复现 / 最终模型播放
uv run train Mjlab-Velocity-Flat-Pikachu-V031-T6 \
  --env.scene.num-envs 4096 --agent.max-iterations 30000 \
  --agent.save-interval 500 --agent.seed 42
uv run play Mjlab-Velocity-Flat-Pikachu-V031-T6 \
  --checkpoint-file logs/rsl_rl/pikachu_v031_velocity/2026-10-10_20-11-53/model_29999.pt \
  --viewer velocity-joystick

# E1
uv run train Mjlab-Velocity-Flat-Pikachu-V031-E1-Clip \
  --env.scene.num-envs 4096 --agent.max-iterations 1001 \
  --agent.save-interval 500 --agent.seed 42 \
  --agent.run-name screen-s42 --agent.logger tensorboard
uv run play Mjlab-Velocity-Flat-Pikachu-V031-E1-Clip \
  --checkpoint-file logs/rsl_rl/pikachu_v031_e1_clip/2026-10-11_21-15-42_screen-s42/model_1000.pt \
  --viewer velocity-joystick

# E2
uv run train Mjlab-Velocity-Flat-Pikachu-V031-E2-Symmetry \
  --env.scene.num-envs 4096 --agent.max-iterations 1001 \
  --agent.save-interval 500 --agent.seed 42 \
  --agent.run-name screen-s42 --agent.logger tensorboard
uv run play Mjlab-Velocity-Flat-Pikachu-V031-E2-Symmetry \
  --checkpoint-file logs/rsl_rl/pikachu_v031_e2_symmetry/2026-10-11_21-29-49_screen-s42/model_1000.pt \
  --viewer velocity-joystick

# E3
uv run train Mjlab-Velocity-Flat-Pikachu-V031-E3-YawHuber \
  --env.scene.num-envs 4096 --agent.max-iterations 1001 \
  --agent.save-interval 500 --agent.seed 42 \
  --agent.run-name screen-s42 --agent.logger tensorboard
uv run play Mjlab-Velocity-Flat-Pikachu-V031-E3-YawHuber \
  --checkpoint-file logs/rsl_rl/pikachu_v031_e3_yaw_huber/2026-10-11_21-45-39_screen-s42/model_1000.pt \
  --viewer velocity-joystick

# E4：本轮 5k 课程覆盖 / 失败边界样本播放
uv run train Mjlab-Velocity-Flat-Pikachu-V031-E4-Curriculum \
  --env.scene.num-envs 4096 --agent.max-iterations 5001 \
  --agent.save-interval 500 --agent.seed 42 \
  --agent.run-name final-s42 --agent.logger tensorboard
uv run play Mjlab-Velocity-Flat-Pikachu-V031-E4-Curriculum \
  --checkpoint-file logs/rsl_rl/pikachu_v031_e4_curriculum/2026-10-11_22-01-49_final-s42/model_5000.pt \
  --viewer velocity-joystick
```

批量训练与评测入口：

```bash
# 完整原设计（E1/E2/E3 5k，E4 30k）
./doc/pikachu_v031_gait_improvement_20261011/run_sequential.sh

# 本报告全部统一遥测、视频和曲线
./doc/pikachu_v031_gait_improvement_20261011/evaluate_results.sh
```

## 验证

- 定向单元测试：21 passed；
- Ruff：通过；
- E1–E4 均完成 64-env/2-iteration preflight；
- E2 额外完成 4096-env capacity preflight；
- 66 组 rollout CSV 已完成列族/NaN 自检；
- 6 段 MP4 均通过 ffprobe 参数校验。
