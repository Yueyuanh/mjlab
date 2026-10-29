# 实验、日志与模型保存规范

本规范适用于新建和继续运行的 mjlab/rsl_rl 实验。历史实验不要求批量迁移；一旦续训、重评或
选为正式候选，应在不移动活跃文件的前提下按本规范归档。

## 设计原则

- **先看阶段，再看 run**：实验根固定为 sanity、smoke、pilot、full 四层，打开目录即可直接进入
  `4-full/`，不在同一层浏览一长串混合 run。
- **run 是原子单元**：一个 run 的配置、checkpoint、曲线、指标、视频和报告放在同一 run 下，复制
  或归档 run 时不依赖实验根其他 artifact 目录。
- **本地证据优先**：W&B 等服务是可选索引，不能代替本地有效配置、TensorBoard、最终指标和
  checkpoint 身份。
- **训练和评估分离**：TensorBoard 说明优化过程；固定协议的 `metrics/` 和 `evaluation/` 才能
  说明最终策略行为。

这些原则综合了 Isaac Lab、rsl_rl、legged_gym、CleanRL 和 Habitat Lab 的实践，调研记录见
[research/open-source-logging.md](research/open-source-logging.md)。

## 唯一路径结构

```text
logs/rsl_rl/<robot>/<task>/<experiment>/
├── README.md
├── 1-sanity/
│   └── <run>/
├── 2-smoke/
│   └── <run>/
├── 3-pilot/
│   └── <run>/
└── 4-full/
    └── <run>/
        ├── config/
        │   ├── agent.yaml
        │   ├── env.yaml
        │   └── run.json
        ├── checkpoints/
        ├── exports/
        ├── tensorboard/
        ├── metrics/
        │   └── README.md
        ├── report/
        │   ├── figures/
        │   └── html/
        ├── evaluation/
        │   ├── video/
        │   └── training_plots/
        └── README.md
```

四个阶段中的 run 统一使用上述内部结构；某阶段没有实际产物的目录可以为空，但不得改回
`training/<run>`、`evaluation/<case>`、实验根 `reports/` 等按 artifact 类型拆分的布局。
`training_plots` 是固定拼写，不使用 `traning_plots`。文件名固定使用 `README.md`，避免在大小写敏感
系统中同时出现两个入口。

`<robot>`、`<task>`、`<experiment>` 使用稳定、可读的 kebab-case 名称：

- `robot`：机器人或硬件版本，例如 `pikachu-v035`；
- `task`：任务语义，例如 `velocity-flat`；
- `experiment`：本轮要验证的假设，例如 `arm-pose-fix`，不要重复日期或写 `test2`、`final-new`；
- 一个 Task ID 的不同实验必须落到不同 `experiment` 目录，不得把多个任务混入同一目录。

实验根 `README.md` 使用[实验入口模板](templates/experiment-readme.md)，只做目标、阶段状态和最终
候选索引，不堆放 run 产物。

## 阶段定义与晋级

| 目录 | 目的 | 最低完成条件 |
| --- | --- | --- |
| `1-sanity/` | zero/random agent、配置、reset、NaN 和维度检查 | 环境可运行，关键不变量通过 |
| `2-smoke/` | 小资源验证 train/save/resume/eval/video 全管线 | 产物结构完整且可恢复 |
| `3-pilot/` | 中等资源验证学习方向和停止条件 | 达到预设 pilot 门槛 |
| `4-full/` | 放大已冻结的 pilot 配置 | 正常结束并完成自动评估与报告 |

不得把失败 run 改名或移动到更高阶段来表示晋级。晋级是新建下一阶段 run，并在两个 run 的
`config/run.json` 中记录父子关系。上一阶段失败时不得直接启动 full；例外必须写入实验 README。

## run 命名

run 名固定为：

```text
E<实验序号>-R<重复批次>-<YY-MM-DD>
```

例如 `E01-R01-26-10-29`：

- `E` 表示该 experiment 下的配置/假设序号，修改有效配置或主要假设时递增；
- `R` 表示同一 `E` 的独立重复批次或 seed，重跑时递增；
- 序号至少两位补零，日期使用启动日期和 `YY-MM-DD`；
- 同一天重启也不得覆盖原 run，必须增加 `R`；续训不改 run 名，并在 `run.json` 写明恢复记录；
- 阶段已经由父目录表达，run 名中不再写 `sanity`、`smoke`、`pilot`、`full`。

## run 内部契约

### `config/`

- `agent.yaml` 和 `env.yaml` 必须是启动时实际生效的解析后配置，不是未经 override 的默认值。
- `run.json` 是单一 manifest，至少记录 schema version、阶段、run ID、Task ID、完整启动命令、seed、
  环境数、迭代目标、开始/结束时间、退出状态、代码 commit 和 dirty 状态、依赖身份、父 checkpoint、
  恢复范围、最终 checkpoint 相对路径和 SHA-256、外部 logger ID/URL。
- dirty 工作树启动正式 run 时，应在 `config/` 保存可复现的 diff，例如 `git.patch`；不得在配置快照
  中写入 token、密码或内网凭据。
- `run.json` 的状态至少区分 `running`、`completed`、`interrupted`、`invalid`。只有
  `completed` 且最终 checkpoint 哈希存在，才表示训练正常结束。

### `checkpoints/`、`exports/` 与 `tensorboard/`

- `checkpoints/` 保存 runner 原生 checkpoint；必须保留足以恢复的 iteration、optimizer 和相关状态。
- 候选选择完成前不得只保留最后一个 checkpoint。不得按编号最大或 mtime 最新自动认定最终模型。
- `exports/` 中每个 ONNX 或其他部署格式都必须在 `run.json` 绑定源 checkpoint SHA-256，并完成一次
  加载/推理一致性检查。
- `tensorboard/` 保存 event 文件；训练 stdout/stderr 可保存为 `tensorboard/train.log`，不要再在实验根
  建立平行的 `logs/training/`。
- 多进程训练仅由 rank 0 写共享日志、配置和 checkpoint，避免 event 与 manifest 冲突。

### `metrics/`

`metrics/` 保存独立评估产生的机器可读时序数据及其同步短视频，不保存 TensorBoard event。CSV、
JSON、MP4 和录制 manifest 使用同一 stem，使 `scripts/plot_logs/serve.py` 能自动关联。

每个 `4-full` run 正常完成后，必须对最终 checkpoint 自动执行两档固定命令评估：

1. 前进：`vx = 0.8 * forward_max`；
2. 后退：`vx = 0.8 * backward_min`，保持负号。

每档均为 10 秒、50 Hz，并保存 CSV、汇总 JSON 和同步 MP4。`forward_max` 与 `backward_min` 必须取自
该 run 的有效 `env.yaml`；禁止在规范或脚本中假定所有任务速度范围相同。文件 stem 必须包含方向、
实际命令速度、80%、时长和采样率，例如：

```text
metrics_forward_vx_p0p40mps_p80_10s_50hz_with_target.csv
metrics_forward_vx_p0p40mps_p80_10s_50hz_with_target.json
metrics_forward_vx_p0p40mps_p80_10s_50hz_with_target.mp4
metrics_backward_vx_m0p24mps_p80_10s_50hz_with_target.csv
metrics_backward_vx_m0p24mps_p80_10s_50hz_with_target.json
metrics_backward_vx_m0p24mps_p80_10s_50hz_with_target.mp4
```

速度编码固定为 `p|m<整数>p<小数>mps`，例如 `+0.40 m/s -> p0p40mps`、
`-0.24 m/s -> m0p24mps`，避免 `-` 被误解为分隔符。若接口生成 `.recording.json`，也使用同 stem。

`metrics/README.md` 使用 [metrics 模板](templates/metrics-readme.md)，至少写明 checkpoint、两档
速度的计算来源、录制命令、字段/单位/坐标系，以及查看方法：

```bash
uv run python scripts/plot_logs/serve.py
```

当前 Pikachu v033 的参考接口是 `scripts/review_v033_policy.py`，其 `--command-x`、`--seconds 10`、
`--video` 和 `--view three_quarter` 可直接完成固定速度数据与视频录制。该脚本现有 crawl 分支仍限制
side view，与本规范冲突；录制 crawl 正式主视频前必须先让接口支持 three-quarter，不得用侧视视频
冒充合规主视频。其他任务应使用等价通用接口；实际参数始终以当前脚本 `--help` 为准。

### `evaluation/`、`report/` 与 run `README.md`

- `evaluation/video/` 保存最终展示视频；`evaluation/training_plots/` 保存训练曲线导出图。
- `report/figures/` 保存报告引用的派生图，`report/html/` 保存可独立打开的 HTML 报告。原始指标仍以
  `metrics/` 为准，重新作图不得改写原始数据。
- run `README.md` 使用 [run 模板](templates/run-readme.md)，必须包含本次实验的简短总结、状态与核心
  指标，嵌入训练曲线和最终效果视频，并给出可复制的训练复现命令、play 命令和带完整参数的视频
  录制命令。
- Markdown 客户端不能内嵌播放 MP4 时，使用带 poster/缩略图的相对链接；不得只贴某台机器的绝对
  路径或只贴 W&B URL。

## 最终视频规格

每个 `4-full` run 的最终效果视频必须满足：

- 1280x720（720p）、50 FPS，使用 mjlab MuJoCo renderer；
- 第三人称斜侧视角（three-quarter），默认 azimuth 45 度、elevation -15 度，跟踪 `base_link`；
- 右下角叠加紧凑速度响应曲线，仅显示 `vx`、`vy`、`wz`，目标虚线、实测实线；
- overlay 与控制步同步，使用 3-5 秒滚动窗口，宽不超过画面 30%，高不超过 32%，边距至少 16 px，
  且不得遮挡机器人；
- 命令与实测使用相同 body/heading frame，`vx/vy` 单位 m/s，`wz` 单位 rad/s；
- 录制 metadata 保存 Task ID、checkpoint SHA-256、命令速度、相机参数、分辨率、FPS、renderer 和
  视频 SHA-256。

该规则统一适用于 locomotion 的四足、爬行、双足和直立策略；任务需要额外视角时可以补录，但斜侧
视角仍是 README 中的主视频。视频只能用于行为复核，不能代替数值评估。

## 活跃 run、续训与清理

- 训练期间 runner 若要求原生布局，必须让 runner 从一开始就把 run 根指向上述 `<stage>/<run>/`；
  不得在进程运行中移动 checkpoint、event 或配置。确实无法兼容时记录例外，并在退出后一次性归档。
- 续训前核对父 checkpoint SHA-256、Task ID、观测/动作契约和 optimizer 加载范围；追加到同一 run
  时必须在 `run.json` 保留每次 continuation 的起止迭代与命令。
- 删除失败 run、重复日志或中间 checkpoint 前，检查 README、报告、续训和最终候选引用，并在实验
  README 记录删除范围、原因和保留的证据。
- `pre_train/` 只接收已选择、已评估的稳定部署 corpus：最终 checkpoint、必要 export、主指标、报告
  和小型配置快照；不得复制整个训练目录。
- “训练完成”“仿真候选”“实机安全审核通过”“可部署”是不同状态，必须显式标注。
