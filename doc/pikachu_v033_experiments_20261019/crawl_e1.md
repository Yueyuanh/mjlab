# V033 crawl_e1 参数与检查记录

2026-10-20，分支 `feat/pikachu-v033-crawl-20261019`。本页对应新任务 **`Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E1`**；独立源码目录 `src/mjlab/tasks/velocity/config/pikachu_v033_crawl_e1/`，默认实验名 `pikahcu_v033/crawl_e1`，默认日志目录 `logs/rsl_rl/pikahcu_v033/crawl_e1/`。

本轮完成配置重设计、物理检查、64环境×5轮PPO和checkpoint录像链路检查。**未进行E1正式长程训练，尚未学会爬行，下面的smoke视频不是最终训练效果。** 原两项3000轮训练及视频仍见 [training_report.md](training_report.md)。

**最新姿态变更（2026-10-20 21:18）：双侧arm_yaw关节数值均改为+90°，对应新增软窗口[85°,95°]。** 已保存的5轮smoke模型/视频仍来自yaw=0°，不是本次yaw=90°姿态的训练结果；本次只做姿态、物理和边界检查。

**最新接触改造（2026-10-20 21:38）：** 手部球形碰撞体已移除，前肢改用原elbow STL mesh；支撑site/摆动高度奖励改为真实表面参考。双足reset悬空和四足debug箭头已修复，详见 [contact_setup_20261020.md](contact_setup_20261020.md)。本页后面的球体阶段检查数据保留为历史，不对应当前接触模型。

## 初态与完整动作

| 关节 | 初态 | 新增软窗口（度） |
|---|---:|---:|
| right_arm_roll | -90° | [-95°, -85°] |
| left_arm_roll | +90° | [+85°, +95°] |
| right_arm_yaw | +90° | [+85°, +95°] |
| left_arm_yaw | +90° | [+85°, +95°] |
| 双侧elbow | 0° | 不纳入此次四关节窗口 |

窗口是奖励的计算区间。现有roll和yaw机械限位均约±90.0002°，与上述窗口相交；yaw的[85°,95°]中超过约90°的部分受原机械限位约束，不代表能对称活动到95°。不修改XML机械限位，不新增±5°硬限位或等式约束。保留机身pitch=1.3rad、root z=0.234402775m、其余腿部和arm_pitch初态；当前前肢mesh表面最低点初始约z=0.013253m，由物理自由落下建立支撑。球体阶段约5.9mm穿地已随球体删除消失。

## 当前前肢接触：原elbow STL mesh

`elbow_link`是刚体，接触形状由附属geom决定。当前Crawl/E1不再添加`*_hand_collision`球，直接启用`right_elbow_link_collision`、`left_elbow_link_collision`原STL mesh，contype/conaffinity均为1。MuJoCo对mesh使用凸包接触；不是逐三角形凹面碰撞。接触传感器仍匹配`*_elbow_link`，原前臂惯性和整机质量不变。

保留`*_hand`作为奖励/滑移测量site（无质量、无碰撞），参考位置为右(0,-.076,-.021)、左(0,+.076,+.021)m，与当前初态下mesh最低表面一致。删除摆动高度奖励旧的15mm球半径扣除项。当前有效接触为10条足底capsule、2个前臂mesh、1个躯干盒。用户注释的ankle STL没有恢复。

历史记录：最初使用半径15mm的末端球以简化计算，21:18的yaw90检查仍采用球体；21:38起按用户要求切换为上述原mesh。历史产物不覆盖。

恢复Crawl及E1完整18维动作，顺序为左右腿各5维，然后左右臂各4维：hip_pitch、hip_roll、hip_yaw、knee_up、ankle；arm_pitch、arm_roll、arm_yaw、elbow。4个被动四杆关节仍由物理机构处理。先前“移除elbow输出、降为16维”的方案已替代，原诊断不覆盖或删去。现在elbow参与策略输出；初态0°不等于训练过程中锁定0°。

动作仍为 `target = default_joint_pos + scale × action`（启用编码器偏置DR时执行器侧再减去偏置）。缩放为hip_pitch=.35，hip_roll/yaw=.15，knee_up/ankle=.25，arm_pitch/roll/elbow=.25，arm_yaw=.20，单位rad/动作单位。动作层clip与runner clip_actions均为None，无新增动作掩码或目标投影。

完整动作接口有利于后续通用策略，但当前双足/四足观测语义、维度仍不同：E1 actor/critic为76/96，双足为74/86。**18维输出一致并不意味着双足checkpoint可直接加载为四足通用策略**；真正共用策略仍需统一观测与任务输入。

## 新增位置软惩罚

仅对表中四个roll/yaw关节，用物理实际角度计算；中心取配置中的默认初态，不随动作目标或观测噪声移动。令 `δ=5°=π/36 rad`：

```text
excess_j = max(abs(q_j - q_init_j) - δ, 0)
cost = mean_j((excess_j / δ)^2)       # 四关节平均
新增奖励 = -1.0 × cost × step_dt     # step_dt=0.02s
```

新增项在闭区间±5°内为0，越界后平方增长，边界处连续且一阶导数为0；只读状态，不改写动作或qpos。实现中的clamp_min仅用于计算非负超出量，不是关节/动作裁剪。原有全身crawl_pose与机械限位惩罚仍保留，因此“窗口内不罚”只指新增项。

| 四关节偏离情况 | cost | 新增加权奖励（按秒计） |
|---|---:|---:|
| 全部≤5° | 0 | 0 |
| 只有1个关节偏10°，其他≤5° | .25 | -.25 |
| 全部偏10° | 1 | -1 |
| 全部偏15° | 4 | -4 |

权重-1.0为首轮设计值，未经过长程训练调参。软惩罚允许策略为速度收益接受越界，不能保证始终保持在±5°；后续评估需统计各关节越界时间占比、峰值偏差与速度/支撑/滑移，避免仅靠总奖励验收。

## 训练参数

继承Crawl工厂，每次创建独立配置；E1只额外增加`arm_posture_window`，双足不受影响。

| 参数 | 任务默认 | 本轮smoke实际 |
|---|---|---|
| 环境数 / 迭代数 | 1024 / 50000 | 64 / 5 |
| 每环境每轮步数 | 24 | 24 |
| 转移样本总数 | 1,228,800,000（跑满默认时） | 7,680 |
| seed | 42 | 42 |
| actor / critic隐藏层 | 各512,256,128，ELU，观测归一化 | 相同 |
| 输入与输出 | actor 76→18，critic 96→1 | 相同 |
| 初始Gaussian标准差 / 熵系数 | .5 / .005 | 相同 |
| PPO epoch / minibatch | 5 / 4 | 相同 |
| 学习率 / schedule | .001 / adaptive，desired KL=.01 | 相同 |
| gamma / lambda / clip | .99 / .95 / .2 | 相同 |
| 梯度范数上限 | 1 | 相同 |
| 保存间隔 | 500 | 5（另保存末轮） |
| 日志 | 独立W&B项目与标签，CLI可切换 | 本地TensorBoard，无W&B上传 |
| 恢复训练 | false | false，从头初始化 |

物理200Hz、控制50Hz、episode 20秒。初期1000轮只采样vx=[.08,.20]m/s、vy/wz=0；随后vx=[-.10,.25]、vy=[-.08,.08]m/s、wz=[-.20,.20]rad/s；10%站立命令。步态周期1.6秒、每肢20%摆动。保留摩擦/编码器偏置/COM±5mm启动随机化，关闭推力；play关闭actor观测噪声与课程。

| 奖励项 | 权重 |
|---|---:|
| track_linear_velocity / linear_velocity_huber | 3 / -.75 |
| track_angular_velocity | 1 |
| crawl_orientation / crawl_height / crawl_pose | -3 / -.3 / -.5 |
| support / gait_contacts | -.5 / 1.5 |
| swing_height / swing_air_time / support_slip | -.5 / .3 / -18 |
| action_rate_l2 / dof_pos_limits / vertical_velocity | -.05 / -2 / -.5 |
| **arm_posture_window（新增）** | **-1** |

## yaw=90°、球体接触检查（历史）

- 双侧yaw均+90°，roll仍为右-90°/左+90°，elbow均0°；原机械限位检查通过，四杆闭环误差约5.8e-9m，质量仍4.7027358kg。
- 18维动作、76/96维观测保持。新增软惩罚自动以新的默认姿态为中心，容差5°、权重-1.0；现有边界测试已使用真实默认姿态，双向±5°及超出部分增长检查通过。
- 4环境×5秒零动作：0跌倒，接触占空比约.996/.996/1/1，roll/yaw峰值偏差约.01052°，新增cost为0。非零动作目标可超过5°，验证无新增目标裁剪。
- 相关17项回归通过，本次文件ruff格式/lint、ty、pyright通过。未重训已有模型，未停止已有Viser进程；需重启播放器加载新初态。
- [几何/碰撞审计](config_audit/crawl_pose_yaw90_20261020.json)；[物理检查](../../logs/rsl_rl/pikahcu_v033/crawl_e1/diagnostics/pose_yaw90_20261020/physics_check.json)。日志/命令/源码留痕位于`crawl_e1/diagnostics/pose_yaw90_20261020/`，原`design_20261020`保留不覆盖。

## 首次设计检查与产物（yaw=0°历史）

- 边界/配置隔离/相关回归：43通过、2跳过；覆盖四关节正负偏离、5°边界、超范围平方增长、elbow不进入该项、奖励不修改关节状态、训练与play保持18动作、独立日志、网络与物理配置保持一致。
- 4环境×5秒零动作：0跌倒，接触占空比约.996/.996/1/1；roll/yaw最大偏差约.00481°，新增cost为0。该检查关闭startup DR与外部辅助，只能说明短时零动作稳定。
- 非零动作目标检查：18路均保持仿射映射，elbow目标确实随策略输出变化；roll偏置目标10.743°、yaw8.594°可超过窗口，证明没有目标裁剪。这是目标映射检查，不是越界物理运动认证。
- 64环境×5轮PPO完成，实际7680转移样本，最终`model_4.pt`；保存env.yaml、agent.yaml、事件和源码差异。
- 新checkpoint前进录像：1环境×5秒，目标vx=.15m/s，实际约.000159m/s，0跌倒；几乎静止，**尚未学会爬行**。
- 本次修改文件ruff格式/lint、ty、pyright通过。全仓既存103项类型问题的历史结论不因此改变，未声明全仓检查通过。

产物路径：

```text
logs/rsl_rl/pikahcu_v033/crawl_e1/diagnostics/
├── 2026-10-20_20-57-24_smoke-design/
│   ├── model_4.pt, model_0.pt, *.onnx, params/, git/, events.*
└── design_20261020/
    ├── physics_check.json, physics_check.log
    ├── tests.log, ruff.log, format.log, ty.log, pyright.log
    ├── train.log, smoke_review.log
    ├── smoke_review/forward_side.mp4, forward_side.json, forward_side.csv
    └── audit.json, commands.jsonl, source_snapshot.tar.gz
```

[物理检查JSON](../../logs/rsl_rl/pikahcu_v033/crawl_e1/diagnostics/design_20261020/physics_check.json) · [smoke视频（非最终效果）](../../logs/rsl_rl/pikahcu_v033/crawl_e1/diagnostics/design_20261020/smoke_review/forward_side.mp4) · [smoke量化结果](../../logs/rsl_rl/pikahcu_v033/crawl_e1/diagnostics/design_20261020/smoke_review/forward_side.json) · [审计/校验值](../../logs/rsl_rl/pikahcu_v033/crawl_e1/diagnostics/design_20261020/audit.json)

## 历史smoke checkpoint回放与当前初态查看

仓库根目录执行，明确指定目前最新的E1 **smoke** checkpoint（仅5轮、训练时yaw=0°）。当前源码yaw已变为90°，动作默认偏置也随之变化；此命令加载旧权重后不等于原训练姿态回放，不作为新姿态效果验证：

```bash
MUJOCO_GL=egl uv run --no-sync play Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E1 \
  --checkpoint-file logs/rsl_rl/pikahcu_v033/crawl_e1/diagnostics/2026-10-20_20-57-24_smoke-design/model_4.pt \
  --num-envs 1 --device cuda:0 --viewer viser
```

只查看初态：

```bash
MUJOCO_GL=egl uv run --no-sync play Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E1 \
  --agent zero --balance False --freeze-joints False \
  --num-envs 1 --device cuda:0 --viewer viser
```

后续首轮3000轮训练命令（本轮未执行；任务工厂默认仍是50000）：

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=4 uv run --no-sync train \
  Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E1 \
  --env.scene.num-envs 1024 --agent.max-iterations 3000 \
  --agent.seed 42 --agent.logger tensorboard --agent.run-name review-crawl_e1 \
  --log-root logs/rsl_rl
```

此命令写入`logs/rsl_rl/pikahcu_v033/crawl_e1/<时间戳>_review-crawl_e1/`，不会覆盖历史Crawl及E1诊断。正式训练后再以各关节越界指标及四种命令视频判断效果。
