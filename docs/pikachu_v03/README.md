# Pikachu v03 记录库

Pikachu **v03**（并联四杆膝，10 驱动关节 / 14 关节）在 mjlab 上的 velocity
行走任务：从 URDF/MJCF 资产接入，到静平衡站姿标定，到训练收敛的完整留痕。

> 与已有 `Pikachu-27dof` / V025（串联膝 14dof）是**两套独立资产**，不要混用。
> v03 的核心差异是膝：只有 `knee_up_joint` 被驱动，`knee_joint` /
> `knee_mid_joint` 是 `<connect>` 约束驱动的被动关节。

## 目录

| 文档 | 内容 |
|---|---|
| [01-model-conversion.md](01-model-conversion.md) | 源 MJCF → 训练 MJCF 的转换，含 4 个必踩的坑 |
| [02-stand-keyframe.md](02-stand-keyframe.md) | 站立 keyframe 标定：从「脚在髋下」到「质心在脚心上方」 |
| [03-task-design.md](03-task-design.md) | velocity 任务设计：动作缩放、奖励、命令空间、课程 |
| [04-worklog.md](04-worklog.md) | 工作日志（按时间倒序） |
| [reports/](reports/) | 训练报告 + [曲线文本快照](reports/training-curve.md) |
| [videos/](videos/) | 最优模型 720P 录像 |

## 快速上手

```bash
cd pikachu_playground/mjlab

# 重建训练 MJCF（改了源模型或胶囊参数后跑）
uv run python scripts/build_v03_train_xml.py

# 重标定静平衡站姿（改了质量/质心后跑）
uv run python scripts/probe_v03_both_legs.py     # 双腿独立解：脚掌俯仰 0° + 脚心 x=0
uv run python scripts/refine_v03_keyframe.py     # 固定点迭代：脚心挪到质心正下方

# 开训前的冒烟测试（分两个进程跑，本机内存不够同时建两个环境）
uv run python scripts/smoke_v03_velocity.py --quiet   # 零动作开环站立，必须过
uv run python scripts/smoke_v03_velocity.py --noisy   # 带 push 扰动，看难度

# 训练
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Velocity-Flat-Pikachu-V03 \
  --env.scene.num-envs 2048 --agent.run-name v03-flat-rNN --video True

# 盯训练（单行，带趋势箭头）
uv run python scripts/v03_watch.py --run <run 目录名>

# 多行看板
uv run python scripts/v03_train_dashboard.py --run <run 目录名>

# ⚠️ 定量评测 / 录像**必须等训练停下来再跑**，否则内存不够会把两边都拖垮
#    （实测 8 秒/帧，训练从 0.72 s/迭代 掉到 30 分钟 6 个迭代，见报告 §6.1）
uv run python scripts/eval_v03_velocity.py \
  --checkpoint logs/rsl_rl/pikachu_v03_velocity/<run>/model_2500.pt
uv run python scripts/record_v03_video.py \
  --checkpoint logs/rsl_rl/pikachu_v03_velocity/<run>/model_2500.pt \
  --out docs/pikachu_v03/videos --tag <run> --clips stand,fwd,back,lat,turn
```

## 关键常量速查

| 量 | 值 | 出处 |
|---|---|---|
| 训练 MJCF | `src/mjlab/asset_zoo/robots/pikachu/xmls/mjcf_v03/pikachu_v03_base_series_train.xml` | `build_v03_train_xml.py` |
| 常量文件 | `src/mjlab/asset_zoo/robots/pikachu/pikachu_v03_constants.py` | — |
| 任务配置 | `src/mjlab/tasks/velocity/config/pikachu_v03/` | — |
| 站立高度 | 0.26728 m（静平衡微屈膝） | `refine_v03_keyframe.py` |
| 质量 | 2.3814 kg | 训练模型实测 |
| 质心 x | -0.02683 m | 静平衡点 |
| 动作维度 | 10 | 无手臂 |

## 当前状态

| 轮次 | 配置 | 结果 |
|---|---|---|
| r01 | 默认 track std + 只有 `alternating_feet_air_time` | ❌ **学出站着不动**：五档实测全 ±0.003、单脚支撑 0.00 |
| r02 | track std 随命令缩放 + 抬脚奖励拆成两项 | ✅ **会走**：前进跟踪比 1.04 / 方向余弦 1.00 / 单脚支撑 0.76 |
| r03 | 给转向让路（yaw 命令占比、pose 放宽、ang std 放宽） | ⏳ 训练中 |

**「机器人能不能稳定行走」已经答了：能**（r02 @ iter 2500，零终止）。
r03 要解决的是 r02 剩下的三条：不会转向（命令 0.30 → 实测 0.005）、
行走时带 0.2~0.33 rad/s 的未受命令偏航、侧移只有 41%。

**当前最优检查点**：`logs/rsl_rl/pikachu_v03_velocity/2026-10-01_12-36-03_v03-flat-r02/model_2500.pt`

### 时间线

- 2026-10-01 资产接入完成，静平衡站姿标定完成，零动作开环站立稳定
  （base_z 0.26728 → 0.26731，|v| → 0.00000，6 s 无终止）。可开训。
- 2026-10-01 r01 训练 500 迭代 → 逐档评测发现**站着不动**，定位到两个独立缺陷。
- 2026-10-01 r02 修复后训练至 2500 迭代 → 评测**确认会走**。
- 2026-10-01 发现评测脚本的命令固定被课程项覆盖（五档结果全同），修复后重测，
  结论不变。
- 2026-10-01 r03 启动（针对转向）。

## 三句话能讲完的坑

1. **`Episode_Reward/track_*` 不能当进度指标** —— 它是 `exp(−err²)` 的逐帧均值，
   由 Jensen 不等式，站桩也能拿 72% 的分。判「走没走」必须用独立于奖励的
   物理量（固定命令档位 + 测净位移）。见报告 §3.4。
2. **课程项会在运行时覆盖你设的命令范围** —— `cfg.curriculum["command_vel"]`
   每步写 `cfg.ranges.*`，stage 0 的 `step=0` 恒成立。评测/录像脚本必须
   `cfg.curriculum.pop("command_vel", None)`。见报告 §6.5。
3. **训练期间不能再起第二个仿真进程** —— 不是 GPU 争用，是内存（swap 已满），
   实测慢 600 倍。见报告 §6.1。
