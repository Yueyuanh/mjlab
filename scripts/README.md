# mjlab/scripts 目录说明

仓库根目录的**辅助脚本**集合——夜间基准、云端训练、演示、工具、容器运行。与 `src/mjlab/scripts/`（CLI 入口：train/play/list-envs 等，已打包为 `uv run` 命令）不同，本目录是**项目级脚本**，不进入发布包。

## 一、目录结构

```
scripts/
├── benchmarks/           # 夜间基准系统
│   ├── README.md         #   使用说明（自含文档）
│   ├── nightly_train.sh  #   夜间基准主脚本（克隆→训练→评估→测吞吐→发报告）
│   ├── measure_throughput.py  # 仿真吞吐量测量
│   ├── generate_report.py     # W&B → HTML 报告生成
│   ├── nightly_images/   #   趋势图示例（velocity_go1 / tracking_g1 / lift_cube_yam）
│   └── systemd/          #   自动化调度
│       ├── mjlab-nightly.timer  # systemd 定时器（每天定时触发）
│       ├── mjlab-nightly.service# systemd 服务单元
│       └── README.md           #   调度配置说明
├── cloud/                # SkyPilot 云端训练（Lambda Cloud）
│   ├── README.md         #   云端训练说明
│   ├── train.yaml        #   训练任务（uv 安装 + 训练）
│   ├── train-docker.yaml #   训练任务（Docker 镜像，可复现）
│   ├── sweep.yaml        #   W&B sweep 配置
│   ├── sweep-cluster.yaml#   sweep 集群定义
│   ├── sweep-agent.yaml  #   sweep agent 任务（每 GPU 一个）
│   └── sweep-launch.sh   #   一键：建 sweep + 开集群 + 每 GPU 一个 agent
├── demos/                # 框架功能演示脚本
│   ├── body_impulse.py           # 身体冲力 + 力可视化
│   ├── contact_sensor_decimation.py # 接触传感器历史捕捉 decimation 漏检的碰撞
│   ├── differential_ik.py        # 差分 IK 交互控制
│   ├── flat_patch_terrain.py     # 平 patch 地形演示
│   └── raycast_sensor.py         # 射线传感器演示
├── tools/                # 开发/文档工具
│   ├── render_terrain_gallery.py # 渲染每种地形类型图片（供文档用）
│   └── terrain_explorer.py       # 交互式单 patch 地形探索器（Viser + MjSpec）
├── run_docker.sh         # 便捷 docker 运行（注入 NVIDIA runtime、挂载、端口）
└── fix_mjpython_macos.sh # macOS 上修复 mjpython（symlink libpython）
```

## 二、各文件作用

### benchmarks/ — 夜间基准

| 文件 | 作用 |
|------|------|
| [nightly_train.sh](benchmarks/nightly_train.sh) | **主脚本**：clone 最新 mjlab → 训练 `Mjlab-Tracking-Flat-Unitree-G1`（4096 envs / 6000 iter / W&B registry 运动）→ 测吞吐 → 生成报告 → 推送到 gh-pages。环境变量：`CUDA_DEVICE`、`WANDB_TAGS`、`SKIP_TRAINING`、`SKIP_THROUGHPUT` |
| [measure_throughput.py](benchmarks/measure_throughput.py) | 测 3 个典型任务（Velocity-Go1 / Tracking-G1 / Lift-Yam）的 `Physics SPS` / `Env SPS` / `Overhead`，结果追加进 `throughput_data.json` |
| [generate_report.py](benchmarks/generate_report.py) | 从 W&B 按 tag（默认 `nightly`）拉历史 run，合并出历史趋势，产出 HTML 报告 |
| systemd/ | `mjlab-nightly.timer` 每天定时触发 `mjlab-nightly.service`（跑 `nightly_train.sh`），实现无人值守 |

### cloud/ — 云端训练（SkyPilot + Lambda）

| 文件 | 作用 |
|------|------|
| [train.yaml](cloud/train.yaml) | SkyPilot 任务：`uv` 安装 mjlab + 跑训练（`TASK`/`NUM_ENVS`/`MAX_ITERATIONS` 通过 `--env` 覆盖） |
| [train-docker.yaml](cloud/train-docker.yaml) | 同上但用 GHCR 预构建 Docker 镜像（可复现环境） |
| sweep*.yaml + sweep-launch.sh | W&B Sweeps 超参搜索：`./scripts/cloud/sweep-launch.sh A100:8` 建 sweep + 开集群 + 每 GPU 一个 agent |

### demos/ — 功能演示

| 文件 | 演示什么 |
|------|----------|
| [body_impulse.py](demos/body_impulse.py) | `envs/mdp/events` 的冲力/推挤 + 力可视化 |
| [contact_sensor_decimation.py](demos/contact_sensor_decimation.py) | `ContactSensor` 的 history 能捕捉到被 decimation 跳过的瞬时碰撞 |
| [differential_ik.py](demos/differential_ik.py) | `DifferentialIKAction` 交互式 IK 控制 |
| [flat_patch_terrain.py](demos/flat_patch_terrain.py) | flat patch 地形生成 |
| [raycast_sensor.py](demos/raycast_sensor.py) | `RayCastSensor` 地形扫描 |

### tools/ — 开发工具

| 文件 | 作用 |
|------|------|
| [render_terrain_gallery.py](tools/render_terrain_gallery.py) | 渲染每种 sub-terrain 类型的图片（供 Sphinx 文档使用） |
| [terrain_explorer.py](tools/terrain_explorer.py) | Viser 交互式地形探索器：逐 patch 查看/调试地形 |

### 其他

| 文件 | 作用 |
|------|------|
| [run_docker.sh](run_docker.sh) | docker 运行便捷包装：注入 `--rm --runtime=nvidia --gpus all`、挂载当前目录到 `/app`、分离 venv、发布 8080 端口、交互模式 |
| [fix_mjpython_macos.sh](fix_mjpython_macos.sh) | macOS 专用：`mjpython` 期望 libpython 在 `.venv/lib/`，创建 symlink 修复；非 macOS 直接跳过 |

## 三、典型用法

```bash
# 夜间基准（完整跑一遍）
./scripts/benchmarks/nightly_train.sh

# 只重新生成报告（跳过训练/吞吐）
SKIP_TRAINING=1 SKIP_THROUGHPUT=1 ./scripts/benchmarks/nightly_train.sh

# 只测吞吐
uv run python scripts/benchmarks/measure_throughput.py --num-envs 4096 --output-dir benchmark_results

# 云端训练（Lambda）
sky launch scripts/cloud/train.yaml --env TASK=Mjlab-Velocity-Flat-Unitree-G1

# 云端超参搜索
./scripts/cloud/sweep-launch.sh A100:8

# 演示
uv run python scripts/demos/raycast_sensor.py

# Docker 运行
./scripts/run_docker.sh

# macOS 修复 mjpython
bash scripts/fix_mjpython_macos.sh
```
