# Plot Logs

这里存放 Pikachu 训练、部署与 Metrics CSV 的浏览器分析工具。

## Metrics CSV Dashboard

`metrics_csv_dashboard.html` 用于查看 Viser Metrics 面板录制的 CSV。页面在浏览器
本地解析数据，不会上传 CSV。

推荐从仓库根目录启动带 CSV 发现接口的只读服务：

```bash
uv run python scripts/plot_logs/serve.py
```

然后打开：

```text
http://localhost:8000/scripts/plot_logs/metrics_csv_dashboard.html
```

页面会递归发现 `logs/**/metrics_*.csv`、按修改时间列出，并自动加载最新文件；
刷新下拉框即可看到运行期间新生成的日志，不需要先下载再拖入。也可以直接双击
HTML 或手动选择/拖放任意 CSV 作为兜底。若 ECharts CDN 无法加载，可在
`scripts/plot_logs/echarts.min.js` 放置 ECharts 5 的浏览器构建文件。

页面支持：

- 自动发现、下拉切换和刷新 `logs/` 内 CSV，也可点击或拖放加载；文件选择器
  优先显示收藏项，收藏可设置别名，侧栏始终显示当前 CSV 的完整路径；
- 按 `joint_position`、`joint_velocity`、`joint_torque`、`base` 等自动分类；
- 同类型曲线叠加，以及「每个关节一行」的左右同名关节对比（行数随 CSV 里实际
  出现的关节数自适应，不写死关节名）；
- 独立的数据对比子界面：固定两份 CSV 后可连续添加多个对比项，每项独立展示；
  覆盖所有共同原始指标及派生加速度、惯量扭矩、机械功率和逐关节四象限 T-N；
- 独立的收藏总结子界面，可一键读取全部可用收藏文件，按指标展示跨文件
  RMS、P95、峰值图和完整统计表；
- 关节 T-N 四象限分析：按机械功率正负区分驱动/制动，支持绝对值折叠、
  电机曲线自动匹配、Q3 对称镜像和输出/吸收机械能汇总；
- 侧栏常驻两张 DM 电机参考图；点击后可在视口内完整放大，支持滚轮缩放、
  放大后拖动图片、标题栏移动窗口、双击/按钮复位以及 Esc 关闭；
- 搜索、左右侧筛选、时间区间筛选、缩放和悬停读数；
- Min、Max、Mean、RMS、绝对峰值和绝对值 P95 统计；
- 导出当前分类总览 PNG、左右对比 PNG，以及当前区间的所选字段 CSV。

Viser 录制的标准 CSV 至少包含：

```text
sample,sim_step,time_s,env_idx,network,<metric columns...>
```

网页也接受额外 Metrics，并会按列名前缀动态分类。

**横轴不必是 `time_s`。** 页面按 `time_s → time → sim_step → step → iteration →
iter → epoch → frame → sample` 的顺序取第一个**数值且单调不减**的列当横轴，
都找不到时退化为样本序号（此时横轴标注为「样本序号」）。所以训练趋势 CSV
（例如 `docs/pikachu_v03/reports/r06b-train-trend.csv`，只有 `iter,sep_cm,…`）
可以直接导入，横轴就是 `Iteration`，其余列全部可绘；被选作横轴的列不会重复出现在
曲线列表里。只有「一个数值列都没有」才会报错。

**关节名不写死。** 任何 `left_/right_<关节>_joint` 形式的列都会被自动发现，
按 `hip_pitch → hip_roll → hip_yaw → knee → ankle → arm_pitch → arm_roll` 的顺序
排列，不在其中的按名字排在后面。所以 pikachu_v03 的膝叫 `knee_up`（而不是
`knee`）时，它照样出现在左右对比、T-N 四象限和惯量面板里。若新机型的某个关节
还需要借用已知关节的电机规格/惯量默认值，在页面脚本的 `JOINT_ALIASES` 里加一条
别名即可（现有：`knee_up → knee`，于是 knee_up 自动用 J4310 规格）。

## Metrics CSV Cli Recording

uv run python -m mjlab.scripts.record_metrics_cli Mjlab-Tracking-Flat-Pikachu-Up \
    --ckpt <model.pt 或含 model_*.pt 的目录> \
    --motion-file motions/motion.npz    # tracking 必需，或任务默认已配置
    --out-dir logs --tag TransitionUp --max-steps 1000


前进

uv run python -m mjlab.scripts.record_metrics_cli Mjlab-Tracking-Flat-Pikachu-Walk --ckpt pre_train/rsl_rl/pikachu_tracking/2026-08-13_17-27-46_trial-pikachu-walk-forward-dual/model_4999.pt --motion-file src/mjlab/mocap/npz/pikachu_walk_npz/pikachu_walk_0.185_0.0_0.0.npz --out-dir pre_train/rsl_rl/pikachu_tracking/2026-08-13_17-27-46_trial-pikachu-walk-forward-dual --tag TransitionUp --max-steps 498


后退

uv run python -m mjlab.scripts.record_metrics_cli Mjlab-Tracking-Flat-Pikachu-Walk --ckpt pre_train/rsl_rl/pikachu_tracking/2026-08-13_18-28-36_trial-pikachu-walk-back-dual/model_4999.pt --motion-file src/mjlab/mocap/npz/pikachu_walk_npz/pikachu_walk_-0.185_0.0_0.0.npz --out-dir pre_train/rsl_rl/pikachu_tracking/2026-08-13_18-28-36_trial-pikachu-walk-back-dual --tag TransitionDown --max-steps 498

## Viser 图片导出目录

在 Viser Metrics 页加载 CSV 后，点击 `Save selected plots + comparisons`。图片会
保存在源 CSV 同级目录；浏览器上传的文件则保存到当前 checkpoint 目录：

```text
images/<CSV文件名>/
├── joint_position/
│   ├── left/
│   ├── right/
│   ├── all_joints_comparison.png
│   └── left_right_comparison_1x7.png
├── joint_velocity/
├── joint_torque/
└── other/
```

单关节图、单侧总览和全关节总览为 1920×1080；1×7 左右对比图为
1920×3780。

## Response CSV Dashboard

`response_csv_dashboard.html` 用于分析 `uv run response <TASK>` 导出的
`response.csv`。它与 `mjlab-response-v1` 长表 schema 对齐，提供：

- 按关节、step/sine 和 case 筛选的位置、误差、速度、扭矩同步波形；
- 上升时间、延迟、超调、调节时间、稳态误差、IAE/ISE 和饱和占比；
- 每个频点未经拟合的原始正弦 command/measured 时域曲线；
- 正弦基波最小二乘拟合、Bode 增益/相位、等效延迟、THD 和 −3 dB 带宽；
- 扭矩–转速散点、峰值/RMS 扭矩、机械功率和可调安全系数选型建议；
- 当前电机型号、限位、PD、关节惯量、armature、减速器与占位/未知参数状态；
- MathJax/LaTeX 渲染的模型推导和当前关节数值代入；
- NaN/缺失值分类诊断，区分未知电机参数、未达到判据和频率覆盖不足。

在同一服务下打开
`http://localhost:8000/scripts/plot_logs/response_csv_dashboard.html`。页面会发现
`logs/response/**/response.csv` 并自动加载最新一次运行。

## Quad Dashboard

`quad_dashboard.html` 是现有的 Quad RL 日志分析页面。它使用自己的后端 API 数据
格式，与 Metrics CSV Dashboard 的纯浏览器 CSV 工作流相互独立。
