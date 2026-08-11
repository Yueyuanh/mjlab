# mjlab/tests 目录说明

本目录是 mjlab 的**单元测试套件**，覆盖框架从 manager 到执行器、传感器、任务、工具的各层。目标是验证"改动代码后行为没有被破坏"，不验证策略学得好不好。

## 一、目录结构

```
tests/
├── conftest.py      # 共享 fixture 和测试工具（见下）
├── smoke_test.py    # 冒烟测试：验证打包产物可导入、可建环境
├── fixtures/        # 测试用微型 MuJoCo XML 模型
│   ├── fixed_base_box.xml           # 固定底座单 box
│   ├── floating_base_box.xml        # 浮动底座单 box（freejoint）
│   ├── fixed_base_articulated.xml   # 固定底座多关节机器人
│   ├── floating_base_articulated.xml# 浮动底座多关节机器人
│   ├── biped.xml                    # 双足机器人 + 地面
│   ├── quadcopter.xml               # 四旋翼（含 rotor 执行器）
│   └── tendon_finger.xml            # 腱驱动手指
└── test_*.py        # 85 个测试文件（分层，见下）
```

**关键设计**：测试不用真实机器人（G1/Go1）跑完整物理，而是用 `fixtures/` 里几十行的微型 XML（一个 box、一个二连杆、一只手指）来测 manager/actuator/sensor 的行为——快、确定、好断言。真实任务配置则用 `test_task_configs.py` 做轻量验证。

## 二、conftest.py 提供的共享设施

| 名称 | 作用 |
|------|------|
| `configure_test_environment` | session 级 autouse fixture，静默 warp（`wp.config.quiet = True`） |
| `get_test_device()` | 返回 CUDA（有卡时）或 CPU；`FORCE_CPU=1` 强制 CPU |
| `fixtures_dir` / `load_fixture_xml()` | 定位/加载 fixtures 下的 XML 内容 |
| `create_entity_with_actuator()` | 从 XML 字符串 + actuator 配置创建 Entity |
| `create_entity_from_fixture()` | 从 fixture 文件创建 Entity（可选挂 actuator） |
| `initialize_entity()` | 编译实体并绑定 Simulation，返回 `(entity, sim)` |
| `make_scene_and_sim()` | 从内联 XML 建 Scene + Simulation 并接好传感器（支持多实体） |
| 7 个 `*_xml` fixture | `fixed_base_box_xml`、`biped_xml` 等加载器 + 2 个内联 XML（`robot_with_floor_xml`、`falling_box_xml`） |

## 三、运行方式

| 命令 | 作用 |
|------|------|
| `make test` / `uv run pytest` | 全套（含 slow） |
| `make test-fast` / `uv run pytest -m "not slow"` | 跳过慢测试（日常迭代） |
| `make test-cpu` | `FORCE_CPU=1 uv run pytest`（验证 CPU-only 行为） |
| `make test-cpu-fast` | CPU + 跳过 slow |
| `uv run pytest tests/<file>.py` | 单个文件 |
| `uv run --isolated --no-project --with dist/*.whl tests/smoke_test.py` | 构建产物冒烟（`make build` 用） |

pytest 配置见 [pyproject.toml](../pyproject.toml) 的 `[tool.pytest.ini_options]`：`slow` marker、`--strict-markers`、过滤一条已知的 MuJoCo MULTICCD 告警。

## 四、85 个测试文件分层

### 1. Manager 层（core API）

| 文件 | 覆盖 |
|------|------|
| `test_action_manager.py` / `test_actions.py` | 动作切分、scale/offset/clip 仿射处理 |
| `test_observation_manager.py` / `test_observation_noise.py` / `test_observation_delay.py` / `test_observation_history.py` / `test_observation_nan_handling.py` | 观测管线：噪声、延迟 buffer、历史堆叠、NaN 策略 |
| `test_command_manager.py` | 指令管理器 |
| `test_rewards.py` | 奖励项加权、`scale_by_dt` |
| `test_terminations.py` | 终止条件 |
| `test_events.py` | 事件 4 种触发模式（reset/step/interval/startup） |
| `test_curriculum_manager.py` / `test_envs_curriculums.py` | 课程管理器 |
| `test_metrics_manager.py` / `test_recorder_manager.py` | 指标 / rollout 录制 |
| `test_manager_config_immutability.py` | 注册任务配置 deepcopy 防串改 |

### 2. 执行器层（actuator）

| 文件 | 覆盖 |
|------|------|
| `test_actuator.py` / `test_pd_actuator.py` / `test_builtin_pd_actuator.py` | PD 位置执行器 |
| `test_dc_actuator.py` / `test_builtin_dcmotor_actuator.py` | DC 电机 |
| `test_learned_actuator.py` | 学习型执行器 |
| `test_delayed_actuator.py` / `test_encoder_bias.py` | 命令延迟、编码器偏置 |
| `test_fused_group.py` / `test_builtin_group.py` / `test_actuator_builtin_group.py` / `test_xml_actuator.py` | 执行器分组、XML 执行器 |
| `test_differential_ik_action.py` | 差分 IK 动作 |

### 3. 传感器层（sensor）

| 文件 | 覆盖 |
|------|------|
| `test_contact_sensor.py`（37KB） | 接触检测：匹配、力、air_time |
| `test_raycast_sensor.py`（40KB） | 射线扫描：Grid/Ring pattern、排除父体 |
| `test_builtin_sensor.py` / `test_projected_gravity_sensor.py` | builtin IMU、重力投影 |
| `test_foot_height_sensor.py` / `test_camera_sensor.py` | 地形高度、相机 |
| `test_sensor_caching.py` | 传感器缓存 |

### 4. 场景 / 实体 / 仿真层

| 文件 | 覆盖 |
|------|------|
| `test_entity.py` / `test_entity_data.py` | 实体索引、状态读写 |
| `test_scene.py` / `test_scene_entity_config.py` / `test_spec_config.py` / `test_spec_utils.py` | 场景、SceneEntityCfg、MjSpec 配置 |
| `test_sim.py` / `test_sim_data.py` | Simulation、Torch↔Warp 桥接 |
| `test_auto_reset.py` | 自动重置逻辑 |
| `test_domain_randomization.py`（76KB，最大） | 域随机化全字段 |

### 5. 任务层

| 文件 | 覆盖 |
|------|------|
| `test_velocity_task.py` / `test_velocity_command.py` / `test_velocity_rewards.py` / `test_velocity_terrain_curriculum.py` | velocity 任务 MDP |
| `test_tracking_task.py` / `test_tracking_metrics.py` | tracking 任务 + MPKPE 指标 |
| `test_manipulation_commands.py` / `test_manipulation_observations.py` | manipulation 任务 |
| `test_task_configs.py` | 所有注册任务的配置可实例化、字段合法 |
| `test_variants.py`（55KB） | 大量配置变体组合 |

### 6. 工具 / 基建 / 查看器

| 文件 | 覆盖 |
|------|------|
| `test_circular_buffer.py` / `test_delay_buffer.py` | 环形 / 延迟 buffer |
| `test_runner.py` / `test_rl_exporter.py` | runner、ONNX 导出 |
| `test_nan_guard.py` | NaN 守卫 |
| `test_terrains.py` / `test_perlin_terrain.py` / `test_terrain_config.py` / `test_terrain_utils.py` / `test_terrain_proportion_spawning.py` / `test_flat_patch_sampling.py` | 地形生成 |
| `test_gpu_selection.py` | GPU 选择 |
| `test_g1_constants.py` / `test_go1_constants.py` | G1/Go1 电机参数推导 |
| `test_viewer_tick.py` / `test_viser_conversions.py` / `test_viser_update_policy.py` / `test_native_viewer_actions.py` / `test_offscreen_renderer.py` / `test_reward_bar_panel.py` / `test_video_recorder.py` | 查看器 / 录制 |
| `test_notebooks.py` | notebook 可执行 |
| `test_tyro_defaults.py` / `test_tyro_flags.py` | tyro CLI 解析 |
| `test_asset_zoo.py` / `test_lab_api_math.py` / `test_xml_utils.py` | 杂项工具 |

## 五、smoke_test.py

独立的冒烟测试：
- 标记 `@pytest.mark.slow`（含在 `make test`，`make test-fast` 跳过）。
- **主要用途是构建验证**：`make build` 用 `uv run --isolated --no-project --with dist/*.whl` 跑它，确认打包出的 wheel/tar.gz 能干净导入 `mjlab`、能创建环境并 close。
- 用 Go1 flat env 在 CPU 上跑，断言 `env.sim.data.time == 0.0`。

## 六、规范约定

来自 [CLAUDE.md](../CLAUDE.md)：
- **PR 前必须 `make test`**（全套通过）；**commit 前必须 `make check`**（format + lint + type，测试不必全跑）。
- 测试用**函数和 fixture**，不用测试类。
- 偏好跑**单个测试文件**而非全套，加快迭代。
- 提交代码必须通过类型检查。
