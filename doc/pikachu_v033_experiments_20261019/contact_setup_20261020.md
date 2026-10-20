# V033 接触、初始站高与 debug arrows

2026-10-20，分支`feat/pikachu-v033-crawl-20261019`。当前改动包含Crawl/E1前肢mesh接触、V033-E0双足reset高度修复、Crawl/E1速度箭头修复。未重训历史checkpoint，历史报告/视频及快照保留。

## 当前碰撞模型

- 删除Crawl生成的左右`*_hand_collision`球，启用原`*_elbow_link_collision` STL mesh（MuJoCo按凸包处理接触）。前臂mesh既参与地面接触，也遵循现有碰撞配对/排除规则。
- `robot.py`中的`SUPPORT_GEOMS`现为`(left|right)_(foot[1-5]|elbow_link)_collision`，前肢接触传感器继续匹配elbow_link。
- 保留`*_hand`无碰撞测量site，位置移到当前初态的mesh支撑表面：右(0,-.076,-.021)、左(0,+.076,+.021)m；摆动高度奖励不再减去15mm球半径。
- 用户注释的左右`ankle_link_collision` STL保持注释，没有重建训练XML。现有足底实际类型为**capsule胶囊体**，不是ellipsoid椭球；每脚5条，总计10条。
- 当前Crawl有效接触体为10条足底capsule、2个elbow mesh和1个躯干盒；整机质量仍4.7027358kg。18维动作、网络规模、双yaw=+90°以及roll/yaw±5°软惩罚保持。

前肢mesh最低点在初始重置高度下为z≈.013253m，后脚底约.003m。保留当前用户关节姿态，随后由物理建立四点支撑；零动作检查前肢接触占空比约99.2%。这是接触/姿态检查，不是新模型训练效果。

## 双足悬空原因与调整位置

V033-E0的基准站高已经贴合足底，但通用velocity环境的`reset_base`继承了额外z偏移`(.01,.05)m`，导致正常训练/策略回放reset后脚底高出地面1–5cm。

现在在`src/mjlab/tasks/velocity/config/pikachu_v033/env_cfgs.py`的V033工厂内覆盖：

```python
cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.0)
```

该配置同时用于训练与play，仅影响V033，不修改通用环境或其他机器人。

| 要调整的量 | 文件与变量 | 当前值 |
|---|---|---|
| 双足基准初始高度 | `src/mjlab/asset_zoo/robots/pikachu/pikachu_v033_constants.py` → `BIPED_KEYFRAME.pos`第三项 | .259412m |
| 每次reset叠加的高度扰动 | `src/mjlab/tasks/velocity/config/pikachu_v033/env_cfgs.py` → `events["reset_base"].params["pose_range"]["z"]` | (0,0)m |
| 双足奖励的目标高度 | 同一`env_cfgs.py` → `STAND_HEIGHT`，供`base_height.target_height`使用 | .259412m |
| 四足基准高度 | `pikachu_v033_crawl/robot.py` → `init_state.pos`第三项 | HEIGHT+.003m |

修改`STAND_HEIGHT`只改变奖励目标，不改变初始物理位置；需要调整双足基准高度时先改`BIPED_KEYFRAME.pos[2]`，并保持奖励目标一致。XML的stand/quad keyframe主要供独立MuJoCo viewer预览，mjlab任务以Python配置为准。

## 四足 debug arrows

此前Crawl把`command.debug_vis`设为False，所以速度命令不参与绘制，Debug Viz内也没有Twist勾选项。当前在Crawl工厂中启用`command.debug_vis=True`，并使用独立`CrawlVelocityCommand`绘制水平heading帧的命令/实际速度；采样、动作、观测和策略网络保持原定义。

Viser路径：**Controls → Scene → Debug Viz → Enabled / Twist**。两者默认开启。蓝色是命令线速度，绿色是命令角速度；青色/浅绿色对应实际速度。线速度箭头保持水平，角速度沿世界Z，不随约74.5°前俯机身旋转。

站立且所有速度为0时，零长度箭头不会显示。零动作查看时可在**Controls → Commands → Twist → Enable**启用速度控制；`--agent zero`会把命令范围设为0，需要先把**Max lin_vel_x**增至例如0.2，再调**lin_vel_x**滑块查看非零命令箭头。该模式下策略动作仍为0，设置命令只用于查看箭头。Debug Viz里的Twist只控制显示，不负责改变命令。需要重启已有播放器加载新配置。

## 验证与留痕

- 34项相关回归通过，2项跳过；本次源码ruff格式/lint、ty、pyright通过。新增测试覆盖75°/95°pitch下水平命令/实际箭头与世界Z角速度方向。
- 双足8环境×10次reset：root z=.259411991m，脚底site高度约1.74–1.89微米（GPU数值误差量级），不再额外悬空1–5cm。
- 四足4环境×5秒零动作：0跌倒，后脚接触占空比约.996/.996，前臂约.992/.992，窗口关节最大偏差约.0527°，新增位置cost=0；奖励与观测均有限值。
- 真实Viser实例在独立8894端口完成setup，确认Debug Viz下Enabled/Twist均存在且为True。非零命令(.15,.03,.1)在reset静止状态生成两个非零命令箭头，线箭头世界Z分量为0；实际速度为0的两条箭头按设计略去。未连接浏览器，未修改用户既有Viser服务。
- 初次GUI诊断误把零速度箭头也计为必须存在，断言失败；修正诊断期望后通过，首次日志保留为`setup_check_attempt1.log`。产品代码无需为零长度箭头造显示。

产物位于`logs/rsl_rl/pikahcu_v033/crawl_e1/diagnostics/mesh_contact_20261020/`：

- [物理检查](../../logs/rsl_rl/pikahcu_v033/crawl_e1/diagnostics/mesh_contact_20261020/physics_check.json)
- [接地/mesh/Viser核验](../../logs/rsl_rl/pikahcu_v033/crawl_e1/diagnostics/mesh_contact_20261020/setup_check.json)
- `verify_setup.py`、`setup_check.log`、`physics_check.log`、各项静态检查/测试日志，以及`audit.json`、`commands.jsonl`和源码快照。

当前初态查看（无外部平衡/关节冻结）：

```bash
MUJOCO_GL=egl uv run --no-sync play Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E1 \
  --agent zero --balance False --freeze-joints False \
  --num-envs 1 --device cuda:0 --viewer viser
```

双足将任务名换为`Mjlab-Velocity-Flat-Pikachu-V033-E0`。原3000轮模型及E1的5轮smoke仍属于旧配置，不作为本次接触改造后训练效果。
