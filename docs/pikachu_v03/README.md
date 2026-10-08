# Pikachu v03 记录库

Pikachu **v03**（并联四杆膝，10 驱动关节 / 14 关节）在 mjlab 上的 velocity
行走任务：从 URDF/MJCF 资产接入，到静平衡站姿标定，到训练收敛的完整留痕。

> 与已有 `Pikachu-27dof` / V025（串联膝 14dof）是**两套独立资产**，不要混用。
> v03 的核心差异是膝：只有 `knee_up_joint` 被驱动，`knee_joint` /
> `knee_mid_joint` 是 `<connect>` 约束驱动的被动关节。

## 目录

| 文档 | 内容 |
|---|---|
| [01-model-conversion.md](01-model-conversion.md) | 源 MJCF → 训练 MJCF 的转换，含 5 个必踩的坑 |
| [02-stand-keyframe.md](02-stand-keyframe.md) | 站立 keyframe 标定：从「脚在髋下」到「质心在脚心上方」 |
| [03-task-design.md](03-task-design.md) | velocity 任务设计：动作缩放、奖励、命令空间、课程 |
| [04-worklog.md](04-worklog.md) | 工作日志（按时间**正序**；被推翻的结论保留原文并就地标注） |
| [reports/](reports/) | 训练报告 + [曲线文本快照](reports/training-curve.md) + [速度追踪报告](reports/velocity-report.html)（HTML/PDF） |
| [videos/](videos/) | 720P 跟踪相机录像（r04-best / r05-15000 / r01-before 三组） |
| [metrics/](metrics/) | 逐档逐关节 metrics CSV（dashboard 可直接加载） |

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

# 关节等效惯量 / 执行器稳定性排查（见 01-model-conversion.md 坑 5）
uv run python scripts/diag_v03_stability.py

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

# 逐档逐关节遥测 -> CSV（列契约与 plot_logs dashboard 一致，可直接拖进去看）
uv run python scripts/record_v03_metrics.py \
  --checkpoint logs/rsl_rl/pikachu_v03_velocity/<run>/model_29999.pt \
  --out docs/pikachu_v03/metrics --tag <run> --seconds 8

# CSV -> HTML 报告 + PDF（headless Chrome；图表 base64 内联，零外部依赖）
uv run python scripts/v03_velocity_report.py \
  --metrics-dir docs/pikachu_v03/metrics --tag <run> \
  --checkpoint logs/rsl_rl/pikachu_v03_velocity/<run>/model_29999.pt \
  --out docs/pikachu_v03/reports/velocity-report.html

# 多轮次对比表（Markdown，可直接贴进报告 §7 的分析）
uv run python scripts/v03_metrics_compare.py \
  --metrics-dir docs/pikachu_v03/metrics \
  --tags r04-best,r05-15000,r06-test --out /tmp/v03_compare.md
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
| r02 | track std 随命令缩放 + 抬脚奖励拆成两项 | ✅ **会走**：前进跟踪比 1.04 / 方向余弦 1.00 |
| r03 | 砍方向桶换 yaw 暴露面（6.79% → 13.49%） | ⚠️ 偏航治好，但后退/侧移退化（**走错了旋钮**） |
| r04 | 方向桶全恢复 + `rel_heading_envs=1.0`（暴露面 23.75%），**跑满 30000 迭代** | ✅ **稳定行走达成** |
| r05 | 加 `feet_lateral_distance`（治「站立时双脚挨太近」），**无门控** | ⚠️ 开度 9.23→16.41 cm 修好了，但**动态垮了**（原地踏步，见下） |
| r06 | 同一项**加命令门控**（只在命令≈0 时生效）+ **从 r04@29999 热启动** | ✅ **门控有效**（r06-test@34998）：动态回到 r04 水平（前进 0.98 / 后退 0.79），走路档开度 18.7→**10.9** cm；全量训练（+25000 迭代）进行中 |

**「机器人能不能稳定行走」已经答完：能。**
r04 @ iter 29999：**前进跟踪比 0.99、方向余弦 1.00、单脚支撑 0.92、
`mean_episode_length` 999.83/1000、`fell_over` 0.000**。

| 能力 | r04 结果 |
|---|---|
| 前进 | ✅ 0.99 / 方向余弦 1.00 |
| 后退 | ✅ 0.81 / 方向余弦 0.99 |
| 侧移 | ⚠️ 0.26（方向对，幅度不足；受 `hip_roll` 限位 ±0.26 制约） |
| 走直线 | ✅ 未受命令偏航 \|base_wz\| < 0.05（r02 是 0.33） |
| 边走边转 | ✅ 走 0.30 m/s 同时转 90°：**5.5 s 到位、残留 1.8°** |
| 原地转 | ❌ 收敛 0.18 —— **训练分布里没有「纯转向」环境**（r05 候选） |

**当前最优检查点**：
`logs/rsl_rl/pikachu_v03_velocity/2026-10-01_14-48-28_v03-flat-r04/model_29999.pt`

### 时间线

- 2026-10-01 资产接入完成，静平衡站姿标定完成，零动作开环站立稳定
  （base_z 0.26728 → 0.26731，|v| → 0.00000，6 s 无终止）。可开训。
- 2026-10-01 r01 训练 500 迭代 → 逐档评测发现**站着不动**，定位到两个独立缺陷。
- 2026-10-01 r02 修复后训练至 2500 迭代 → 评测**确认会走**。
- 2026-10-01 发现评测脚本的命令固定被课程项覆盖（五档结果全同），修复后重测，
  结论不变。
- 2026-10-01 r03 启动（针对转向）→ 终评：偏航治好，但后退/侧移退化。
- 2026-10-01 r04 启动（方向桶全恢复 + `rel_heading_envs=1.0`）→ **跑满 30000 迭代**。
- 2026-10-01 修正 `单脚支撑` 指标（原版量的是接触点数不是脚数）。
- 2026-10-08 r04 终评 + 检查点扫描 + 转向测试（发现前后两次转向评测**都问错了
  问题**，见报告 §5.8）。
- 2026-10-08 交付 720P 录像（11 个 mp4，r04-best 六档 + r01-before 对照），
  见 [videos/README.md](videos/README.md)。
- 2026-10-08 收尾整理记录库，改回一条**从未被现实反驳过**的错误归因
  （模型转换期以为「源模型自己发散」，其实一直是 mjlab 默认的 `implicitfast`
  积分器在挡），见 [01-model-conversion.md](01-model-conversion.md) 坑 5。
- 2026-10-08 r05 启动（新增 `feet_lateral_distance`，治「站立时双脚挨太近」）。
- 2026-10-08 录像改**跟踪相机** + 5 秒，加 `--views {side,front}`；查出「看不清」
  的真根因是取景（旧自由相机只占 12% 画面高），见
  [videos/README.md](videos/README.md)。
- 2026-10-09 r05 停训（15320/30000）→ 终评：**开度修好了**（站立 9.23 → 16.41 cm），
  但**动态垮了** —— 逐项对比发现它拿线速度换了角速度（`track_linear_velocity`
  0.72→0.38，`error_vel_xy` 0.63→1.06，而 `error_vel_yaw` 0.77→0.39）。
  根因：这一项**没有命令门控**，走路时也在把 `hip_roll` 往外推，
  和步态直接对撞。见 [04-worklog.md](04-worklog.md)。
- 2026-10-09 r06 启动：同一项**加命令门控**（只在命令≈0 的 env 生效，
  走路 env 恒为 0）+ **从 r04@29999 热启动**（字面意义的「在 r4 的基础上」）。
- 2026-10-09 r06 测试段（5000 迭代，到 34998）终评 —— **门控假设成立**：
  后退档跟踪比从 r05 的 −0.34（**反向**）拉回 0.79，前进 0.95~0.98，
  与 r04 持平；而走路档开度从 r05 的 18.7 cm 回到 **10.9** cm（r04 是 10.1），
  站档 14.0 cm。**代价**：峰值扭矩占用从 r04 的 0.70 变成 **1.00**（打满）。
  见 [reports/analysis-notes.md](reports/analysis-notes.md)。
- 2026-10-09 r06 全量训练启动（从 34998 续 25000 迭代，目标 59998）。
  踩到「`model_34999.pt` 不存在」——`learn(N)` 跑的是 `range(start, start+N)`，
  终点是 `start+N−1`。用软链补齐文件名，见 [04-worklog.md](04-worklog.md)。

## 三句话能讲完的坑

1. **`Episode_Reward/track_*` 不能当进度指标** —— 它是 `exp(−err²)` 的逐帧均值，
   由 Jensen 不等式，站桩也能拿 72% 的分。判「走没走」必须用独立于奖励的
   物理量（固定命令档位 + 测净位移）。见报告 §3.4。
2. **课程项会在运行时覆盖你设的命令范围** —— `cfg.curriculum["command_vel"]`
   每步写 `cfg.ranges.*`，stage 0 的 `step=0` 恒成立。评测/录像脚本必须
   `cfg.curriculum.pop("command_vel", None)`。见报告 §6.5。
3. **训练期间不能再起第二个仿真进程** —— 不是 GPU 争用，是内存（swap 已满），
   实测慢 600 倍。见报告 §6.1。
4. **默认值会掩盖你的归因** —— 两次栽在同一个模式上：mjlab 默认的
   `implicitfast` 积分器掩盖了源模型的稳定性问题（坑 5），`ViewerConfig` 默认的
   `origin_type=AUTO` 掩盖了「相机根本没跟踪」（它长得就像「自动跟踪」，
   实际被离屏渲染器当自由相机）。读默认值要读**消费它的那段代码**。
   见 [videos/README.md](videos/README.md)。
5. **看一个奖励项，要看它借道哪个执行器** —— r05 的 `feet_lateral_distance`
   罚的是「双脚横向间距」，但它是靠推 `hip_roll` 实现的，而 `hip_roll`
   同时是侧向平衡和转向的执行器。加项时只算了「站立时偏差多大、罚多重」，
   漏算了「走路时这个执行器还要干别的活」。**结果就是拿线速度换角速度**
   （`track_linear_velocity` 掉一半、`error_vel_yaw` 反而减半）。
   修法是**给这一项加命令门控**，而不是调小权重 —— 调权重会同时削弱
   它该起的作用。见 [04-worklog.md](04-worklog.md)。
