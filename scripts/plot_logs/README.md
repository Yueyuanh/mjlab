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

- 自动发现、下拉切换和刷新 `logs/` 内 CSV，也可点击或拖放加载；
- 按 `joint_position`、`joint_velocity`、`joint_torque`、`base` 等自动分类；
- 同类型曲线叠加，以及固定 1 列 7 行的左右同名关节对比；
- 搜索、左右侧筛选、时间区间筛选、缩放和悬停读数；
- Min、Max、Mean、RMS、绝对峰值和绝对值 P95 统计；
- 导出当前分类总览 PNG、左右 1×7 PNG，以及当前区间的所选字段 CSV。

Viser 录制的标准 CSV 至少包含：

```text
sample,sim_step,time_s,env_idx,network,<metric columns...>
```

`time_s` 是必需列。网页也接受额外 Metrics，并会按列名前缀动态分类。

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
