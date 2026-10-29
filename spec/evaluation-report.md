# 评估与报告规范

## 评估协议

评估必须在运行前固化机器可读协议，至少包含：

- Task ID、checkpoint 相对路径与 SHA-256、代码和有效配置身份；
- 推理模式（确定性均值动作或采样）、seed、环境数、时长、warmup 和控制 dt；
- 命令/动作序列、初始状态、终止处理、随机化开关与参数；
- 指标定义、单位、坐标系、聚合方式和成功/失败阈值；
- renderer、相机、分辨率、FPS，以及是否启用调试可视化。

最终评估应当同时包含 nominal 和 randomized/robustness 条件。比较 baseline 与 candidate 时，
必须用同一任务构建、同一动力学、同一观测、同一命令和同一评估代码；否则标为跨协议参考，
不得直接排序。

训练期 TensorBoard/W&B 曲线不能代替独立评估。均值也不能掩盖振荡、跌倒和长尾；关键指标
应同时给出离散度、分位数或逐 rollout 数据。

## 最低产物

每个正式 case 至少保存：

- 原始或足够重算结论的时序数据（CSV/NPZ/JSON）；
- 汇总 JSON，包含协议与指标；
- 日志及退出状态；
- 需要人工观察时的 MP4、contact sheet 和 render metadata；
- 产物哈希或 manifest。

机器可读数据是证据源，图和表是派生物。重新作图不得改写原始评估数据。

产物必须保存在对应 run 内：原始时序、汇总 JSON 和同步短视频进入 `metrics/`；最终展示视频进入
`evaluation/video/`；训练曲线进入 `evaluation/training_plots/`；派生图和 HTML 报告分别进入
`report/figures/` 与 `report/html/`。完整目录和命名规则见
[experiment-logging.md](experiment-logging.md)。

## locomotion 视频

- 使用 mjlab `ManagerBasedRlEnv(render_mode="rgb_array")` 和 MuJoCo renderer。
- 所有 locomotion 最终主视频统一为 1280x720、50 FPS、第三人称斜侧视角：azimuth 45°、
  elevation -15°、跟踪 `base_link`。四足/爬行和双足/直立使用相同主视角。
- 任务需要侧视、正视或其他诊断视角时可以补录，但不能替代 README 中的斜侧主视频。
- 速度 overlay 只显示 `vx`、`vy`、`wz`；命令虚线、实测实线，按控制步同步。
- overlay 使用 3-5 秒滚动窗口，宽不超过画面 30%，高不超过 32%，边距至少 16 px，不能挡住机器人。
- 命令与实测必须使用相同的 body/heading frame；`vx/vy` 单位 m/s，`wz` 单位 rad/s。
- 保存 renderer、相机、分辨率、FPS、Task ID、命令速度、checkpoint SHA-256、指标文件与视频
  SHA-256。

每个 full run 还必须自动录制前进和后退最大速度绝对值的 80%，各 10 秒、50 Hz，并保存同 stem
CSV/JSON/MP4。速度计算、文件名和 dashboard 查看方式以
[experiment-logging.md](experiment-logging.md) 为准。

视频用于检查行为和沟通，不能替代数值评估；精选片段不能冒充全量成功率。

## 报告结构

正式报告应按以下顺序组织：

1. 结论与当前状态：通过、未通过、仿真候选或可部署；
2. 评估对象与 checkpoint 身份；
3. 协议和与基线的可比性；
4. 核心指标、波动和失败案例；
5. 图、视频和原始数据链接；
6. 训练变量与因果限制；
7. 已知风险、未验证项和下一步。

报告必须保留退化和失败尝试，区分观察事实、推断和建议。涉及实机时，仿真结果不得表述为
安全证明；必须另列 actuator 余量、速度/力矩/温度限制和实机审核状态。
