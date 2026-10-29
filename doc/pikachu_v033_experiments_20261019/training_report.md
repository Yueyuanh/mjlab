# V033 双足基线与四足爬行训练审核报告

状态：complete；开始时间：2026-10-19T23:01:37.889995+08:00；更新时间取自 status.json。

分支：`feat/pikachu-v033-crawl-20261019`。原工作区有大量既存修改，已保存 provenance/preexisting.patch / provenance/preexisting-status.txt；没有覆盖或提交这些既存修改。

## 配置检查

E0：4.702736 kg，22 个关节，其中 18 驱动 / 4 被动四杆关节；50 Hz 策略 / 200 Hz 物理；左右镜像增强与 mirror loss=0.05。
J4310 额定配置 12.5 N·m，J3507 3.0 N·m。此处为仿真扭矩限幅，未建模完整电机转速-扭矩曲线或热限制。
原四足 QUAD_KEYFRAME 为双足占位，不能直接训练；新任务使用独立配置包、独立初态、前肢球形接触、四通道接触观测、世界竖直 yaw reset、水平 heading 速度观测/奖励和前俯姿态终止。
E0 保留原训练配方；其仅脚底接触模式不模拟手臂/躯干触地和大部分自碰撞，必须在部署前补充更完整碰撞与硬件验证。
源 XML 左右 ankle 限位同为[-0.79,1.05]、被动 knee 同为[-0.65,1.05]，而镜像符号表对应关节反号；非对称限位在反号映射下不保持不变。本轮保留来源，应核对真实机械限位。
已确认 E0 接触顺序为右/左，高度扫描却为左/右；feet_swing_height 直接逐列组合两者，存在左右配对错误。foot_clearance 也将模型顺序的 site 速度与扫描高度组合。Crawl 使用显式一致的右后/左后/右前/左前顺序，无此扫描混配。E0 本轮作为原始配置基线保留，详见 config_audit/e0_sensor_order_audit.json。
原 E0 默认 num_envs=1，不适合直接全量启动；本轮显式设置并行数。原任务默认 50000 轮，本轮为有限预算的首轮训练，不能声称 50k 收敛结果。
双足双臂8关节参与策略动作，行走pose容差较宽，standing_posture_l2仅在近零命令生效；本轮未单独验收摆臂幅度、速度与能耗。最终录像为确定性策略，持续摆臂不能归因于录像时随机探索，也不能未经对照就判为合理自然摆臂。

## 实际训练参数

| 参数 | 双足 E0 | 四足 Crawl |
|---|---|---|
| 迭代预算 | 3000 | 3000 |
| 并行环境 | 1024 | 1024 |
| 转移样本预算 | 73,728,000 | 73,728,000 |
| 随机种子 | 42 | 42 |
| 每轮步数 / batch | 24 / N×24 | 24 / N×24 |
| actor / critic MLP | 512,256,128; ELU; obs normalization | 同左 |
| actor / critic / action 维度 | 74 / 86 / 18 | 76 / 96 / 18（含2维节律相位） |
| PPO epochs / minibatches | 5 / 4 | 5 / 4 |
| 学习率 / 调度 | 0.001 / adaptive KL=0.01 | 同左 |
| γ / GAE λ / clip / grad norm | 0.99 / 0.95 / 0.2 / 1.0 | 同左 |
| 熵系数 / 初始动作 std | 0.01 / 1.0 | 0.005 / 0.5 |
| 镜像增强 / mirror loss | 开 / 0.05 | 关，避免误用双足接触布局 |
| checkpoint | 每 500 轮 + 结束 | 同左 |
| 命令 x / y / yaw | 初始 [-.4,.5]/[-.35,.35]/[-.3,.3]；2000轮后[-.6,.8]/[-.5,.5]/[-.5,.5] | 前1000轮[.08,.20]/0/0；之后[-.1,.25]/[-.08,.08]/[-.2,.2] |
| 静止样本 | 10% | 10% |
| 爬行节律 | 无 | 周期1.6秒，每肢20%摆动、至多一肢摆动；抬高目标2cm |
| 四足改进 | — | 跟踪std=.08m/s，Huber归一化尺度=.10m/s、权重-.75，节律接触奖励1.5 |
| 扰动 | 原 E0 摩擦/COM/编码器/推力 | 摩擦/编码器/COM±5mm，无推力 |
| actor 线/角速度均匀噪声 | ±0.5 m/s / ±0.2 rad/s | ±0.05 m/s / ±0.10 rad/s |
| 动作硬裁剪 | 无；位置执行器及物理限位约束 | 同左 |

单张 RTX 4060 Laptop 8 GB，顺序运行以避免争用。完整实际参数以每项 run 的 params/env.yaml 与 params/agent.yaml 为准。TensorBoard 本地记录，不向外部平台上传。

## 实际奖励权重

来自本轮已写出的有效 env.yaml；尚未启动的任务列留空。

| 奖励项 | 双足 E0 | 四足 Crawl |
|---|---|---|
| action_acc_l2 | -0.02 | — |
| action_rate_l2 | -0.15 | -0.05 |
| air_time | 0.5 | — |
| alternating_air_time | 3.0 | — |
| angular_momentum | -0.02 | — |
| base_height | -0.2 | — |
| base_vertical_velocity | -0.5 | — |
| body_ang_vel | -0.05 | — |
| crawl_height | — | -0.3 |
| crawl_orientation | — | -3 |
| crawl_pose | — | -0.5 |
| dof_pos_limits | -1.0 | -2 |
| flight | -2.0 | — |
| foot_clearance | -4.0 | — |
| foot_slip | -0.1 | — |
| foot_swing_height | -2.0 | — |
| gait_contacts | — | 1.5 |
| linear_velocity_huber | -0.25 | -0.75 |
| pose | 1.0 | — |
| self_collisions | -1.0 | — |
| short_step_air_time | -2.0 | — |
| soft_landing | -0.002 | — |
| standing_posture_l2 | -0.25 | — |
| support | — | -0.5 |
| support_slip | — | -18.0 |
| swing_air_time | — | 0.3 |
| swing_height | — | -0.5 |
| torque_soft_limit | -0.25 | — |
| track_angular_velocity | 2.0 | 1 |
| track_linear_velocity | 2.0 | 3 |
| upright | 1.0 | — |
| vertical_velocity | — | -0.5 |
| walking_stance_width | -0.2 | — |

## 实测结果

以下评估固定 seed=42，关闭 startup 域随机化与推力，每命令 32 环境×12 秒；这是标称评估，不代表鲁棒性或多种子统计。跌倒自动 reset，视频保留 reset，MAE 含 reset 过渡。
接触滑移列使用接触期间支撑 site 的世界水平速度作为代理；尤其球形前肢接触可能包含滚动，不等价于真实接触点的表面滑移。占空比顺序见配套JSON的contact_order。

| 实验 / 命令 | 实测 vx/vy/wz | MAE vx/vy/wz | 无跌倒比例 | 总跌倒 | 接触占空比 | 接触时滑移速度m/s |
|---|---|---|---|---|---|---|
| biped/backward | -0.276/0.000/-0.002 | 0.024/0.043/0.072 | 100.0% | 0 | 0.568/0.572 | 0.059/0.058 |
| biped/forward | 0.397/0.000/0.008 | 0.021/0.031/0.096 | 100.0% | 0 | 0.543/0.548 | 0.035/0.037 |
| biped/stand | 0.001/0.003/0.004 | 0.010/0.040/0.060 | 100.0% | 0 | 0.595/0.609 | 0.027/0.028 |
| biped/turn | 0.005/0.000/0.262 | 0.013/0.043/0.078 | 100.0% | 0 | 0.571/0.597 | 0.036/0.031 |
| crawl/backward | -0.066/0.008/0.007 | 0.019/0.014/0.061 | 100.0% | 0 | 0.761/0.708/0.730/0.590 | 0.017/0.012/0.024/0.013 |
| crawl/forward | 0.143/0.001/0.037 | 0.010/0.006/0.050 | 100.0% | 0 | 0.656/0.583/0.772/0.633 | 0.013/0.025/0.016/0.025 |
| crawl/stand | -0.001/0.003/-0.004 | 0.001/0.003/0.009 | 100.0% | 0 | 1.000/0.998/0.998/0.998 | 0.003/0.001/0.002/0.003 |
| crawl/turn | -0.017/0.000/0.078 | 0.017/0.009/0.072 | 100.0% | 0 | 0.918/0.736/0.870/0.881 | 0.010/0.007/0.012/0.022 |

biped 状态：complete；checkpoint：`/home/finnox/Pikachu/PikachuRobot/pikachu_playground/mjlab/logs/rsl_rl/pikachu_v033/biped_e0/2026-10-19_23-01-42_review-biped/training/main/checkpoints/model_2999.pt`。
完成时间（含评估/录像）：2026-10-20T00:20:53.327264+08:00；SHA256：`3f720ea0f2935fe28fc3aacab57aba67b35ec384f3315c9be0ba5ba2b513eded`。

backward：三点及以上支撑比例=63.3%，完全腾空比例=0.2%。
forward：三点及以上支撑比例=47.9%，完全腾空比例=0.2%。
stand：三点及以上支撑比例=99.8%，完全腾空比例=0.0%。
turn：三点及以上支撑比例=84.8%，完全腾空比例=0.2%。

crawl 状态：complete；checkpoint：`/home/finnox/Pikachu/PikachuRobot/pikachu_playground/mjlab/logs/rsl_rl/pikachu_v033/crawl/2026-10-20_00-20-57_review-crawl/training/main/checkpoints/model_2999.pt`。
完成时间（含评估/录像）：2026-10-20T00:48:10.721899+08:00；SHA256：`69aacc9d70e699edbe327a2a52e61cfacc32718207b9d8d1978f73ad4fd4cf41`。

## 审核关注点

- biped 静止命令的接触检测占空比为 0.595/0.609；需要审核原地踏步或接触抖动。
- 四足前进虽能移动，三点及以上支撑仅 47.9%，尚不能按严格三点支撑慢爬验收。
- crawl 原地转向不足：命令 0.150rad/s，实际均值 0.078rad/s，MAE 0.072rad/s。

## 视频与复现

流程完成后，视频使用最终实际 checkpoint 的确定性 actor，侧视和正视均为 1280×720、50fps、8 秒，包含静止、前进、后退、转向；没有筛除失败镜头。未完成时下面只列已实际产出的文件。
- [biped/backward_front](../../logs/rsl_rl/pikachu_v033/biped_e0/2026-10-19_23-01-42_review-biped/evaluation/review/videos/backward_front.mp4)
- [biped/backward_side](../../logs/rsl_rl/pikachu_v033/biped_e0/2026-10-19_23-01-42_review-biped/evaluation/review/videos/backward_side.mp4)
- [biped/forward_front](../../logs/rsl_rl/pikachu_v033/biped_e0/2026-10-19_23-01-42_review-biped/evaluation/review/videos/forward_front.mp4)
- [biped/forward_side](../../logs/rsl_rl/pikachu_v033/biped_e0/2026-10-19_23-01-42_review-biped/evaluation/review/videos/forward_side.mp4)
- [biped/review_all](../../logs/rsl_rl/pikachu_v033/biped_e0/2026-10-19_23-01-42_review-biped/evaluation/review/videos/review_all.mp4)
- [biped/stand_front](../../logs/rsl_rl/pikachu_v033/biped_e0/2026-10-19_23-01-42_review-biped/evaluation/review/videos/stand_front.mp4)
- [biped/stand_side](../../logs/rsl_rl/pikachu_v033/biped_e0/2026-10-19_23-01-42_review-biped/evaluation/review/videos/stand_side.mp4)
- [biped/turn_front](../../logs/rsl_rl/pikachu_v033/biped_e0/2026-10-19_23-01-42_review-biped/evaluation/review/videos/turn_front.mp4)
- [biped/turn_side](../../logs/rsl_rl/pikachu_v033/biped_e0/2026-10-19_23-01-42_review-biped/evaluation/review/videos/turn_side.mp4)
- [crawl/backward_front](../../logs/rsl_rl/pikachu_v033/crawl/2026-10-20_00-20-57_review-crawl/evaluation/review/videos/backward_front.mp4)
- [crawl/backward_side](../../logs/rsl_rl/pikachu_v033/crawl/2026-10-20_00-20-57_review-crawl/evaluation/review/videos/backward_side.mp4)
- [crawl/forward_front](../../logs/rsl_rl/pikachu_v033/crawl/2026-10-20_00-20-57_review-crawl/evaluation/review/videos/forward_front.mp4)
- [crawl/forward_side](../../logs/rsl_rl/pikachu_v033/crawl/2026-10-20_00-20-57_review-crawl/evaluation/review/videos/forward_side.mp4)
- [crawl/review_all](../../logs/rsl_rl/pikachu_v033/crawl/2026-10-20_00-20-57_review-crawl/evaluation/review/videos/review_all.mp4)
- [crawl/stand_front](../../logs/rsl_rl/pikachu_v033/crawl/2026-10-20_00-20-57_review-crawl/evaluation/review/videos/stand_front.mp4)
- [crawl/stand_side](../../logs/rsl_rl/pikachu_v033/crawl/2026-10-20_00-20-57_review-crawl/evaluation/review/videos/stand_side.mp4)
- [crawl/turn_front](../../logs/rsl_rl/pikachu_v033/crawl/2026-10-20_00-20-57_review-crawl/evaluation/review/videos/turn_front.mp4)
- [crawl/turn_side](../../logs/rsl_rl/pikachu_v033/crawl/2026-10-20_00-20-57_review-crawl/evaluation/review/videos/turn_side.mp4)

命令原文与状态在 `/home/finnox/Pikachu/PikachuRobot/pikachu_playground/mjlab/logs/rsl_rl/pikachu_v033/_audit/20261019`；正式训练日志在各运行目录的 train.log，评估/录像日志为 eval.log、video.log。历史命令中的旧路径由 migration_manifest.json 映射至新位置。
[目录与固定 checkpoint 回放指令](README.md)；原始训练参数 YAML 保留原文，experiment_name 反映迁移前名称。

不能只凭奖励曲线判定学会四足：需要人工确认四肢确实交替支撑、没有拖肘滑行、没有持续穿模，结合速度误差与生存率审核。若退化或未收敛，本轮应标记不通过并据此继续调参。

[训练曲线](../../logs/rsl_rl/pikachu_v033/_audit/20261019/reports/figures/training_curves.png) · [曲线数据](../../logs/rsl_rl/pikachu_v033/_audit/20261019/reports/figures/training_curves.csv) · [工作记录](WORKLOG.md)
两任务的奖励定义不同，曲线仅用于查看各自训练过程，不能按奖励数值直接排名。
[机读参数审计](config_audit/asset_audit.json) 包含实际编译的质量、关节限位、执行器参数与动作缩放。

验证：本次新增任务/脚本通过 ruff、ty、pyright；相关回归34通过/2跳过。全仓 make check -o format 剩余103项既有类型诊断，详见日志；没有把它记为全仓通过。
