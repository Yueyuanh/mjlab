# Plot Logs

这里存放 Pikachu 训练、部署与 Metrics CSV 的浏览器分析工具。

## Metrics CSV Dashboard

`metrics_csv_dashboard.html` 用于查看 Viser Metrics 面板录制的 CSV。页面在浏览器
本地解析数据，不会上传 CSV。

推荐从仓库根目录启动：

```bash
python -m http.server 8000 -d scripts/plot_logs
```

然后打开：

```text
http://localhost:8000/metrics_csv_dashboard.html
```

也可以直接双击 HTML；若 ECharts CDN 无法加载，建议改用上面的本地 HTTP 服务，
并在 `scripts/plot_logs/echarts.min.js` 放置 ECharts 5 的浏览器构建文件。

页面支持：

- 点击或拖放加载 CSV；
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

## Quad Dashboard

`quad_dashboard.html` 是现有的 Quad RL 日志分析页面。它使用自己的后端 API 数据
格式，与 Metrics CSV Dashboard 的纯浏览器 CSV 工作流相互独立。
