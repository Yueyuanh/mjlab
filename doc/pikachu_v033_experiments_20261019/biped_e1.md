# V033 直立 E1：手臂范围、臂腿避碰与静止

2026-10-20，分支 `feat/pikachu-v033-crawl-20261019`。独立任务：
`Mjlab-Velocity-Flat-Pikachu-V033-E1`；默认日志：
`logs/rsl_rl/pikahcu_v033/biped_e1/`。

## 初态和手臂约束

初始角度写在 `src/mjlab/asset_zoo/robots/pikachu/pikachu_v033_constants.py`
的 **BIPED_KEYFRAME.joint_pos**。腿部未列出的关节默认0；root高度仍为
0.259412m，reset额外z偏移仍为(0,0)。

| 关节 | 左初态 | 右初态 | 左活动窗口 | 右活动窗口 |
|---|---:|---:|---|---|
| arm_pitch | 0° | 0° | [-10°,10°] | [-10°,10°] |
| arm_roll | -53° | 53° | [-53°,-45°] | [45°,53°] |
| arm_yaw | 0° | 0° | [-5°,5°] | [-5°,5°] |
| elbow | 0° | 0° | [-10°,0°] | [0°,10°] |

`|45,53|`按左右镜像的绝对幅度区间理解。双侧pitch/yaw范围关于0对称，
roll和elbow范围严格左右反号，因此继续保留原镜像增强/损失0.05。

奖励在 `pikachu_v033_biped_e1/env_cfgs.py` 中的 `ARM_BOUNDS_DEGREES`
配置，函数在同包 `mdp.py::arm_position_window`。对实际关节位置q计算：

```text
excess_i = max(lower_i - q_i, 0) + max(q_i - upper_i, 0)
cost = mean_i[(excess_i / ((upper_i-lower_i)/2))²]  # 8个臂关节
reward = -2.0 × cost × dt
```

窗口内此项为0；窗口外连续增长。没有裁剪输出/位置目标，没有移除elbow，
没有新增关节硬限位。roll初态位于窗口边缘，重力和惯性导致的偏差会产生代价，
策略需要主动补偿；软奖励不保证永不越界。

共享BIPED_KEYFRAME的当前E0也会采用此手臂初态；E0原奖励/碰撞配置保留。
历史E0 checkpoint、视频、参数YAML、源码快照不改写，不能将当前代码回放
误记为原始训练配方的精确复现。Crawl使用独立初态，不随本次keyframe变化。

## 臂腿不碰撞：调研与实施

原E0的 `FEET_ONLY_COLLISION` 会清零所有非足底geom的碰撞掩码。
虽然配置中已有self_collisions奖励，臂腿STL根本不生成接触，奖励无从发现穿插。
仅收窄臂关节范围也不足以覆盖抬腿、侧移、转向时的相对几何变化。

E1采用两层约束：

1. 启用原8个臂mesh与12个腿部mesh，脚底仍为10条胶囊。碰撞掩码只允许
   **臂↔腿/足底**以及原足底支撑接触；保留原相邻机构exclude，避免四杆连杆和
   肩部装配的正常相邻几何被错误地当成需要避开的自碰撞。
2. 采用MuJoCo 3.11的 `margin=0, gap=.01m`：表面距离小于10mm时生成
   不施加接触力的检测记录，提前施加平方距离不足代价；实际接触/穿透时由正常
   接触求解器产生分离力，并通过4个物理子步的力历史惩罚短时碰撞。

| 对象 | contype | conaffinity |
|---|---:|---:|
| 地面 | 1 | 1 |
| 臂mesh | 2 | 4 |
| 腿mesh | 4 | 2 |
| 足底胶囊 | 5 | 3 |

MuJoCo的配对条件是 `(type1 & affinity2) || (type2 & affinity1)`。
臂不与地面接触、腿部mesh不新增地面支撑，避免策略把手臂/膝盖当成额外支撑；
用户注释的ankle STL不恢复。`arm_leg_contact`每个臂body保留最近接触，
同时覆盖同侧和对侧腿。当前不把臂↔躯干、臂↔臂或腿↔腿纳入本项研究范围。

接近代价为 `sum[(max(1-dist/.01,0))² × found]`，权重-2。
碰撞力代价权重-2，超过1N的子步计数。普通STL接触使用**凸包**，对凹槽可能
比真实CAD保守；若后续误报明显，可采用离线凸分解。对于大批量训练，贴合几何的
胶囊/凸体代理更快，但需与CAD校验覆盖，不能任意缩成小球来制造间隙。

本轮直接读取官方MuJoCo文档源码并核对已安装MuJoCo/Warp实现，避免混用旧版
margin/gap语义。原始参考材料保存在诊断目录的 `references/` 中：

- [碰撞选择与margin/gap](https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection)
- [geom碰撞掩码、margin和gap](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom)
- [官方文档源码](https://github.com/google-deepmind/mujoco/blob/main/doc/XMLreference.rst)

CPU和真实GPU传感器均已确认：+8.42mm的未接触状态可以触发接近代价；
-35.54mm人为穿透探针能触发接触和较大惩罚。探针只检查检测链路，没有将其
当成有效步态。标称腿位下256组窗口内臂姿态均未进入10mm缓冲区；2048组全范围
臂姿态中187组进入缓冲区/接触。有限抽样不能证明所有行走配置均不会碰撞。

## 静止命令的优化

站立判据统一为 `norm(cmd_xy)+abs(cmd_yaw) <= .05`。
纯转向命令也属于运动，不能仅检查xy速度。步态奖励在大于.05时启用；
静止时没有air_time等抬脚奖励，抑制原地踱步不应靠给抬脚奖励加分实现。

| 配置/奖励 | E1设置 |
|---|---|
| standing样本比例 | .10 → .25 |
| standing_posture_l2 | 权重-.25 → -1；目标为完整BIPED_KEYFRAME |
| 站立姿态std，单位rad | hip_pitch/knee_up/ankle .06；hip_roll/hip_yaw .05；arm_pitch .08；arm_roll .05；arm_yaw .04；elbow .05 |
| standing_joint_velocity | 权重-.5；18驱动关节速度平方均值，std=1rad/s |
| standing_base_motion | 权重-.5；log1p(sum((线速度/.05)²)+sum((角速度/.15)²)) |
| standing_foot_motion | 权重-1；log1p(两脚三维速度平方均值/.03²) |
| standing_support | 权重-3；缺少支撑的脚数/2 |
| alive / fall | 权重+1 / -2500；dt=.02，非超时跌倒单次-50 |
| 行走时pose阈值 | .05，避免低速行走被当成紧站姿 |
| 外部push | 只对运动命令环境生效；静止不随机踢动 |
| COM随机化 | 原±25/25/30mm收为三轴±5mm |

这些静止项以命令门控，行走/转向时关闭；未关闭正常步态奖励。
E1将足高扫描与接触传感器统一为right,left，连同foot_clearance速度列和
foot_swing_height落地列一并对齐，修复E0中的左右混配问题。

**“零动作”不等于训练后的静止控制。** 新初态以固定PD目标运行4环境×5秒出现
24次失稳终止；没有通过冻结关节或外部balance掩盖。静止需要策略持续补偿重力
和微小扰动，不能承诺关节速度数学上恒为0。本轮的可审核候选标准是：
8环境×10秒、0跌倒；略过首秒后base速度RMS<.01m/s、足部速度RMS<.005m/s、
双支撑比例≥99%。同时单独报告手臂越界比例、最大越界和臂腿接触，不用速度均值
代替姿态/接触验收。

## 模型与训练参数

| 参数 | 值 |
|---|---|
| actor / critic输入 | 74 / 86，与E0一致 |
| 动作输出 | 18，全部驱动关节 |
| actor / critic隐藏层 | 512 / 256 / 128，ELU |
| 控制/物理 | 50Hz / 200Hz，dt=.02，decimation=4 |
| PPO rollout | 每环境24步 |
| PPO epochs / minibatches | 5 / 4 |
| clip / gamma / lambda | .2 / .99 / .95 |
| entropy / KL / grad norm | .01 / .01 / 1 |
| learning rate | 默认1e-3自适应；本轮适应pilot从3e-4开始 |
| symmetry | 镜像增强及mirror loss=.05 |
| 默认最大迭代/保存间隔 | 50000 / 500，非本轮实际训练量 |
| W&B project | pikachu-v033-velocity-biped-e1 |

5轮smoke：64环境×24×5=7680转移，本地TensorBoard，无外部上传。
短程适应从原E0的model_2999.pt仅载入actor及其归一化器；critic、optimizer、
迭代计数和课程重新初始化。每轮的参数YAML、初始化SHA、原始训练日志各自保留。
初版300轮pilot在训练中出现回合缩短，因此保留并调整为上述robust motion及
fall惩罚后重新启动500轮，不能将初版当作已通过的静止改进。

几何、零动作失败、回归与训练证据位于
`logs/rsl_rl/pikahcu_v033/biped_e1/diagnostics/design_20261020/`。

## 查看初态

在mjlab目录执行，关闭播放器的外部平衡及冻结：

```bash
MUJOCO_GL=egl uv run --no-sync play Mjlab-Velocity-Flat-Pikachu-V033-E1 \
  --agent zero --balance False --freeze-joints False \
  --num-envs 1 --device cuda:0 --viewer viser
```

暂停后查看初始角度；继续运行零动作会失稳，该命令不是训练效果展示。
已有Viser进程没有被中断，需要重启加载新代码。

## 本轮训练与效果核验

最新候选：**512环境×500轮×24步=6,144,000转移，seed42**，
2026-10-20 22:13启动、约5分钟完成。只载入旧E0 actor/normalizer，
没有继承旧critic、optimizer或课程计数。它属于短程适应候选，未自动当成正式
全向行走模型，也没有覆盖旧E0 checkpoint。

| 静止检查 | 0跌倒 | base速度RMS | 足速度RMS | 双支撑 | 静止候选验收 |
|---|---|---:|---:|---:|---|
| 旧E0 actor直接放入当前E1，8环境×10秒 | 是 | 168.74mm/s | 238.26mm/s | 18.56% | 未通过 |
| 初版300轮，8环境×10秒 | 否，24次终止 | 388.51mm/s | 212.20mm/s | 92.67% | 未通过 |
| 修订500轮，8环境×10秒 | 是 | **0.218mm/s** | **0.767mm/s** | **100%** | 通过 |
| 修订500轮，16环境×30秒，seed123及默认startup随机化 | 是 | **0.132mm/s** | **0.438mm/s** | **100%** | 每环境均通过 |

表中速度/支撑略过首秒，跌倒计数包含完整rollout。标称8环境固定相同初态且关闭
startup随机化；随机化16环境保留摩擦、COM和编码器偏置，随机世界yaw/平移，
每环境分别验收。随机化最差环境base/足速度RMS为.194/.587mm/s，均低于阈值。
名义静止手臂无窗口越界；随机化静止峰值越界约.0055°。全部静止检查无臂腿接触。
名义静止从初始姿态到平衡过程的最大base水平位移约16.5mm，不能把首秒也说成
完全不动；没有使用外部平衡、冻结关节或强写qpos。

**行走限制保留给人工审核：** 前进目标vx=.4m/s，首秒后世界系平均速度约
(.3555,.0761,.00525)m/s，有侧向漂移；8环境×10秒0跌倒、未检出臂腿接近/
穿透。前进时pitch、roll、elbow最大越界分别约3.47°、1.96°、2.70°，yaw无
越界，说明软约束显著收敛但不等于硬边界。转向目标wz=.3rad/s，实测约
**.0030rad/s**，基本站住，**转向未通过**；失败录像同样保留。当前候选的静止
改善可审核，全向控制仍需后续针对转向的训练。

证据：

- [当前配置及全部奖励参数](../../logs/rsl_rl/pikahcu_v033/biped_e1/diagnostics/design_20261020/design_config.json)
- [几何/GPU接近检测及零动作失稳](../../logs/rsl_rl/pikahcu_v033/biped_e1/diagnostics/design_20261020/physics_check.json)
- [旧actor在新环境中的对照](../../logs/rsl_rl/pikahcu_v033/biped_e1/diagnostics/design_20261020/e0_actor_in_e1.json)
- [初版pilot失败评估](../../logs/rsl_rl/pikahcu_v033/biped_e1/diagnostics/design_20261020/pilot_v1_eval.json)
- [当前候选静止/前进/转向评估](../../logs/rsl_rl/pikahcu_v033/biped_e1/diagnostics/design_20261020/pilot_v2_eval.json)
- [16环境随机化静止及每环境结果](../../logs/rsl_rl/pikahcu_v033/biped_e1/diagnostics/design_20261020/pilot_v2_randomized_stand.json)
- [源码/产物校验与审计](../../logs/rsl_rl/pikahcu_v033/biped_e1/diagnostics/design_20261020/audit.json)

相关回归56通过/2跳过；本次10个Python文件ruff格式/lint、ty、pyright通过。
原始逐步测量另存`.traces.pt`，命令、失败日志、参数YAML和源码快照均保留。

### 最新checkpoint和回放

```bash
MUJOCO_GL=egl uv run --no-sync play Mjlab-Velocity-Flat-Pikachu-V033-E1 \
  --checkpoint-file logs/rsl_rl/pikahcu_v033/biped_e1/diagnostics/2026-10-20_22-13-28_pilot-v2-robust-standing/model_499.pt \
  --balance False --freeze-joints False \
  --num-envs 1 --device cuda:0 --viewer viser
```

在Controls→Commands→Twist中启用控制并将vx/vy/wz均设为0检查静止。
可直接观看下列每段5秒的720p/50fps候选录像；视频另用单环境生成，与8/16环境
量化rollout分开留痕，不能把其中一段当成所有条件均通过的证明。

| 录像 | 内容 |
|---|---|
| [静止正面](../../logs/rsl_rl/pikahcu_v033/biped_e1/diagnostics/2026-10-20_22-13-28_pilot-v2-robust-standing/review/stand_front.mp4) | 检查双臂姿态和双脚静止 |
| [静止侧面](../../logs/rsl_rl/pikahcu_v033/biped_e1/diagnostics/2026-10-20_22-13-28_pilot-v2-robust-standing/review/stand_side.mp4) | 检查首秒平衡过程及后续静止 |
| [前进侧面](../../logs/rsl_rl/pikahcu_v033/biped_e1/diagnostics/2026-10-20_22-13-28_pilot-v2-robust-standing/review/forward_side.mp4) | 检查臂腿间隙、步态和漂移 |
| [转向侧面（未通过）](../../logs/rsl_rl/pikahcu_v033/biped_e1/diagnostics/2026-10-20_22-13-28_pilot-v2-robust-standing/review/turn_side.mp4) | 保留基本站住的失败结果 |
