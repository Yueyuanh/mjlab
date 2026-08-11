# mjlab/src/mjlab 源码目录说明

mjlab 框架本体。基于 **MuJoCo Warp**（GPU 物理）+ **Isaac Lab 风格 manager-based API** 的机器人强化学习训练库。

## 一、顶层结构

```
src/mjlab/
├── __init__.py           # 包初始化：EGL 渲染、Warp 配置、任务自动发现
├── py.typed              # PEP 561 标记（本包提供类型信息）
├── actuator/             # 执行器模型（PD / DC / learned / builtin / XML）
├── asset_zoo/            # 机器人资产（G1 / Go1 / i2rt Yam）
├── entity/               # 机器人实体抽象（索引、状态读写、变体）
├── envs/                 # 环境本体 + mdp 原子函数库
├── managers/             # ★ manager-based 架构核心（9 大 manager）
├── rl/                   # RL 层：runner / config / vecenv 包装 / 导出
├── scene/                # 场景组装（实体 + 传感器 + 地形 → MjSpec）
├── scripts/              # CLI 入口与工具脚本
├── sensor/               # 传感器（接触 / 射线 / 高度 / 相机 / builtin）
├── sim/                  # 仿真封装（mjwarp Model/Data + 随机化）
├── tasks/                # 任务注册表 + 具体任务
├── terrains/             # 地形生成（primitive / heightfield / generator）
├── utils/                # 通用工具（buffers / noise / spec / wandb 等）
└── viewer/               # 可视化（native / viser / offscreen）
```

## 二、启动初始化（`__init__.py`）

导入 `mjlab` 时依次执行：

1. Linux 上默认设置 `MUJOCO_GL=egl`（在 mujoco 导入前，保证 GPU 离屏渲染）。
2. `_configure_warp()`：`wp.config.enable_backward = False`，可选 `MJLAB_WARP_QUIET` 静默。
3. `_configure_mediapy()`：把 mediapy 指向 imageio-ffmpeg。
4. `_import_registered_packages()`：加载 entry-point 组 `mjlab.tasks` 注册的外部任务包。

同时定义 `MJLAB_SRC_PATH` 与 tyro 全局 flag（`AvoidSubcommands`、`FlagConversionOff` 等）。

## 三、各子模块详解

### managers/ — 架构核心

9 大 Manager 由 `ManagerBasedRlEnvCfg` 的各字段驱动：

| 文件 | Manager | 职责 |
|------|---------|------|
| [event_manager.py](managers/event_manager.py) | `EventManager` | 域随机化 + 状态重置，4 种触发模式：`reset`/`step`/`interval`/`startup` |
| [command_manager.py](managers/command_manager.py) | `CommandManager` | 目标/参考生成（速度指令、参考运动等） |
| [action_manager.py](managers/action_manager.py) | `ActionManager` | 动作张量切分、scale/offset/clip，下发给执行器 |
| [observation_manager.py](managers/observation_manager.py) | `ObservationManager` | 观测拼接、噪声、裁剪、缩放、延迟、历史堆叠 |
| [termination_manager.py](managers/termination_manager.py) | `TerminationManager` | 终止条件与超时判断 |
| [reward_manager.py](managers/reward_manager.py) | `RewardManager` | 加权求和奖励项，默认按 `dt` 缩放 |
| [curriculum_manager.py](managers/curriculum_manager.py) | `CurriculumManager` | 课程学习（地形难度、指令范围递进） |
| [metrics_manager.py](managers/metrics_manager.py) | `MetricsManager` | 自定义每步指标（按 episode 平均记录） |
| [recorder_manager.py](managers/recorder_manager.py) | `RecorderManager` | rollout 数据记录，无配置时零开销 no-op |
| [manager_base.py](managers/manager_base.py) | `Manager`/`ManagerBase` | 所有 manager 的基类 |
| [scene_entity_config.py](managers/scene_entity_config.py) | `SceneEntityCfg` | 按正则匹配实体/关节/几何的配置 |

### envs/ — 环境本体 + mdp 函数库

| 文件 | 作用 |
|------|------|
| [manager_based_rl_env.py](envs/manager_based_rl_env.py) | **核心环境** `ManagerBasedRlEnv`：`load_managers()` 按固定顺序实例化各 manager，`step()` 实现 decimation 物理步 + 管理链路 + auto-reset |
| [types.py](envs/types.py) | `VecEnvObs` / `VecEnvStepReturn` 类型别名 |
| [mdp/](envs/mdp/) | **原子函数库**（任务只负责用 `func=mdp.xxx` 组合） |

`envs/mdp/` 内容：

| 文件 | 作用 |
|------|------|
| [actions/](envs/mdp/actions/) | 动作 term：`actions.py`（JointPosition/Velocity/Effort、Tendon、Site）、`differential_ik.py`（差分 IK） |
| [dr/](envs/mdp/dr/) | 域随机化函数：`_core`、`_types`、`actuator`、`body`、`camera`、`geom`、`joint`、`light`、`material`、`pair`、`site`、`tendon` |
| [events.py](envs/mdp/events.py) | 状态重置/推挤等通用事件 |
| [observations.py](envs/mdp/observations.py) | 通用观测函数（`joint_pos_rel`、`projected_gravity`、`height_scan` 等） |
| [rewards.py](envs/mdp/rewards.py) | 通用奖励函数 |
| [terminations.py](envs/mdp/terminations.py) | 通用终止函数（`time_out`、`bad_orientation` 等） |
| [metrics.py](envs/mdp/metrics.py) | 通用指标函数 |
| [curriculums.py](envs/mdp/curriculums.py) | 通用课程函数 |

### rl/ — RL 层

| 文件 | 作用 |
|------|------|
| [runner.py](rl/runner.py) | `MjlabOnPolicyRunner`（rsl_rl `OnPolicyRunner` 子类）：checkpoint 迁移、`common_step_counter` 持久化、ONNX 导出 |
| [config.py](rl/config.py) | `RslRlModelCfg` / `RslRlPpoAlgorithmCfg` / `RslRlBaseRunnerCfg` / `RslRlOnPolicyRunnerCfg` |
| [vecenv_wrapper.py](rl/vecenv_wrapper.py) | `RslRlVecEnvWrapper`：ManagerBasedRlEnv → rsl_rl VecEnv，TensorDict obs，dones 合并 |
| [exporter_utils.py](rl/exporter_utils.py) | ONNX 导出辅助（metadata attach 等） |
| [spatial_softmax.py](rl/spatial_softmax.py) | spatial softmax 层（视觉策略用） |

### sim/ — 仿真

| 文件 | 作用 |
|------|------|
| [sim.py](sim/sim.py) | `Simulation`：封装 mjwarp Model/Data，求解器/积分器映射，CUDA graph 捕获 |
| [sim_data.py](sim/sim_data.py) | Torch ↔ Warp 桥接，把 mjwarp kernel 状态映射为 torch 张量 |
| [randomization.py](sim/randomization.py) | `expand_model_fields`：按 EventManager 声明扩展可随机化字段 |

### scene/ / entity/ — 场景与实体

| 文件 | 作用 |
|------|------|
| [scene/scene.py](scene/scene.py) | `SceneCfg`/`Scene`：从内置 scene.xml MjSpec 编译，添加地形/实体/传感器，支持 `write()` 导出 |
| [entity/entity.py](entity/entity.py) | `EntityCfg`/`Entity`：`EntityIndexing`（全局 id/地址映射）、状态读写 |
| [entity/data.py](entity/data.py) | `EntityData`：关节/体/接触等 batch 张量维护 |
| [entity/variants.py](entity/variants.py) | 逐 world 网格变体：同一 batched 场景不同并行世界用不同 mesh，需同一运动学拓扑 |

### actuator/ — 执行器

| 文件 | 作用 |
|------|------|
| [actuator.py](actuator/actuator.py) | `TransmissionType`（JOINT/TENDON/SITE）、`ActuatorCfg`（armature/frictionloss/delay） |
| [pd_actuator.py](actuator/pd_actuator.py) | PD 位置控制执行器（stiffness/damping/effort_limit） |
| [dc_actuator.py](actuator/dc_actuator.py) | DC 电机动力学模型 |
| [learned_actuator.py](actuator/learned_actuator.py) | 学习型执行器（学响应模型） |
| [builtin_actuator.py](actuator/builtin_actuator.py) / [xml_actuator.py](actuator/xml_actuator.py) | 原生/XML 执行器封装 |
| [builtin_group.py](actuator/builtin_group.py) / [fused_group.py](actuator/fused_group.py) | 执行器分组管理 |

### sensor/ — 传感器

| 文件 | 作用 |
|------|------|
| [sensor.py](sensor/sensor.py) | `Sensor` 基类 |
| [contact_sensor.py](sensor/contact_sensor.py) | 接触检测（足端触地、自碰撞），支持 force 历史与 `track_air_time` |
| [raycast_sensor.py](sensor/raycast_sensor.py) | 地形扫描（`GridPattern`/`RingPattern`，排除父体） |
| [terrain_height_sensor.py](sensor/terrain_height_sensor.py) | 指定点阵的地形高度探测 |
| [camera_sensor.py](sensor/camera_sensor.py) | RGB/深度相机（操作任务用） |
| [builtin_sensor.py](sensor/builtin_sensor.py) | 原生传感器（IMU 角/线速度、角动量、重力投影等） |
| [sensor_context.py](sensor/sensor_context.py) | 传感器 sense 图管理 |

### terrains/ — 地形

| 文件 | 作用 |
|------|------|
| [terrain_generator.py](terrains/terrain_generator.py) | 程序化地形组合 |
| [primitive_terrains.py](terrains/primitive_terrains.py) | box 类地形（阶梯、随机块、窄梁） |
| [heightfield_terrains.py](terrains/heightfield_terrains.py) | 高度场地形（Perlin、斜坡、波浪） |
| [terrain_entity.py](terrains/terrain_entity.py) | `TerrainEntityCfg`：地形实体配置 |
| [config.py](terrains/config.py) | `ROUGH_TERRAINS_CFG` 等预设 |
| [utils.py](terrains/utils.py) | 地形工具函数 |

### asset_zoo/ — 机器人资产

| 文件 | 作用 |
|------|------|
| [robots/unitree_g1/g1_constants.py](asset_zoo/robots/unitree_g1/g1_constants.py) | G1：电机型号 → 反射惯量 → stiffness/damping → `G1_ACTION_SCALE`，HOME/KNEES_BENT 关键帧，碰撞配置 |
| [robots/unitree_go1/go1_constants.py](asset_zoo/robots/unitree_go1/go1_constants.py) | Go1 同理 |
| [robots/i2rt_yam/yam_constants.py](asset_zoo/robots/i2rt_yam/yam_constants.py) | i2rt Yam（双臂操作机器人） |
| 各 `__init__.py` | 导出 `get_<robot>_robot_cfg()` 返回 `EntityCfg` |

### tasks/ — 任务系统

| 文件 | 作用 |
|------|------|
| [registry.py](tasks/registry.py) | `register_mjlab_task` / `list_tasks` / `load_env_cfg` / `load_rl_cfg` / `load_runner_cls`，写入 `_REGISTRY` |
| [__init__.py](tasks/__init__.py) | 通过 `utils/lab_api/tasks/importer.py` 递归自动导入所有子包 |

具体任务（velocity / tracking / manipulation / cartpole）见各子目录，均为同构模板：

```
tasks/<name>/
├── <name>_env_cfg.py   # 基础工厂 make_*_env_cfg()：声明式组装全部 MDP
├── mdp/                # 本任务特有原子函数（commands/observations/rewards/terminations/...）
├── rl/runner.py        # 任务特有 runner（如 VelocityOnPolicyRunner）
├── config/<robot>/     # 机器人级配置：__init__.py 注册任务、env_cfgs.py 定制、rl_cfg.py
└── scripts/            # 任务专用脚本（如 tracking 的 evaluate.py）
```

### scripts/ — CLI 入口

见 [scripts/](../scripts/README.md)（仓库根脚本目录）与 `src/mjlab/scripts/` 下的 CLI 实现：

| 文件 | 作用 |
|------|------|
| [train.py](scripts/train.py) | 训练入口（`uv run train <TASK>`），单卡直跑 / 多卡 torchrunx |
| [play.py](scripts/play.py) | 评估/播放入口（`uv run play <TASK>`） |
| [demo.py](scripts/demo.py) | 下载预训练 checkpoint + 运动文件播放 demo |
| [csv_to_npz.py](scripts/csv_to_npz.py) | 运动数据预处理：CSV → 播放 FK → NPZ |
| [list_envs.py](scripts/list_envs.py) | 列出所有已注册任务 ID |
| [visualize_terrain.py](scripts/visualize_terrain.py) | Viser 交互式地形可视化 |
| [nan_viz.py](scripts/nan_viz.py) | 用 Viser 回放 NaN dump（npz + mjb） |
| [export_scene.py](scripts/export_scene.py) | 导出场景为目录/zip（XML + 网格） |
| [gcs.py](scripts/gcs.py) | Google Cloud Storage 下载工具 |
| [_cli.py](scripts/_cli.py) | CLI 共享 helper |

### viewer/ — 可视化

| 文件 | 作用 |
|------|------|
| [base.py](viewer/base.py) | 查看器基类 |
| [native/](viewer/native/) | MuJoCo 原生查看器（viewer.py / visualizer.py / keys.py） |
| [viser/](viewer/viser/) | Viser Web 查看器：`camera_viewer`、`scene`、`viewer`、`overlays`、`term_plotter`（奖励实时曲线）、`reward_bar_panel` |
| [offscreen_renderer.py](viewer/offscreen_renderer.py) | 离屏渲染（录视频用） |
| [debug_visualizer.py](viewer/debug_visualizer.py) | Debug 可视化接口（manager/sensor 各自实现 `debug_vis`） |
| [model_sync.py](viewer/model_sync.py) | 逐 world 模型字段同步进宿主 MjModel |
| [viewer_config.py](viewer/config.py) | `ViewerConfig` |

### utils/ — 通用工具

| 文件 | 作用 |
|------|------|
| [gpu.py](utils/gpu.py) | GPU 选择与管理（`select_gpus`） |
| [torch.py](utils/torch.py) | torch 后端配置（TF32 / deterministic） |
| [os.py](utils/os.py) | `dump_yaml`、checkpoint 路径查找（本地 / W&B） |
| [wandb.py](utils/wandb.py) | W&B 工具（`add_wandb_tags` 等） |
| [nan_guard.py](utils/nan_guard.py) | NaN 守卫：捕捉 NaN 状态并 dump（npz+mjb） |
| [spec.py](utils/spec.py) | MjSpec 工具 |
| [spec_config.py](utils/spec_config.py) | `SpecCfg`/`TextureCfg`：按正则匹配 geom 的属性配置 |
| [xml.py](utils/xml.py) | `MjSpec.to_xml()` 输出修复 |
| [mujoco.py](utils/mujoco.py) | 执行器类型判断、dof 宽度等 |
| [actuator.py](utils/actuator.py) | Electric actuator 工具 |
| [random.py](utils/random.py) | `seed_rng` 随机种子 |
| [logging.py](utils/logging.py) | 彩色终端日志 |
| [spaces.py](utils/spaces.py) | 轻量观测/动作空间定义 |
| [string.py](utils/string.py) | 正则表达式解析（`resolve_expr` / `filter_exp` / `resolve_field`） |
| [color.py](utils/color.py) | RGB/HSV 颜色转换 |
| [lab_api/](utils/lab_api/) | 来自 Isaac Lab 的数学/导入工具（`tasks/importer.py` 递归导入、`math` 四元数运算） |

## 四、核心数据流速览

```
任务配置 → Scene(编译 MjSpec) → Simulation(mjwarp) → load_managers()(9 大 manager)
                                                    ↓
        step(action): process_action → decimation×物理步 → terminations
                      → rewards → events → auto_reset → commands → observations
                                                    ↓
        RslRlVecEnvWrapper → MjlabOnPolicyRunner → PPO → model_<iter>.pt + policy.onnx
```

详情见仓库根 [Architecture.md](../../../Architecture.md) 或 Sphinx 文档 `docs/source/`。
