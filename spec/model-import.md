# 模型导入与格式转换规范

## 输入与来源

- 必须保留原始 URDF/MJCF、被引用 mesh、机器人配置及转换所需文件的只读快照。
- 必须生成 source manifest，至少记录源路径或来源标识、每个输入文件 SHA-256、转换工具与命令。
- 源文件和训练派生文件必须分开命名；不得手工覆盖源模型后仍沿用旧哈希。
- 应当记录模型版本、坐标系、单位、关节零位、轴方向、左右镜像规则和电机数据来源。
- 不导入备份目录、临时导出、模拟器日志及未被模型引用的 mesh。

## 转换实现

- 必须使用可重复执行的脚本和 XML/URDF 解析器完成转换，不使用不可审计的文本替换。
- 脚本应当幂等：对同一输入重复执行，输出物理语义和 manifest 不漂移。
- 所有主动修改必须在 manifest 的 `transformations` 中说明，例如 freejoint、传感器、碰撞体、
  keyframe、关节重排或训练专用 actuator 处理。
- 依赖数组顺序的转换必须按关节或 body 名称重映射 qpos/qvel/keyframe；不得只重排 XML 节点。
- 生成的训练资产必须能由仓库内相对路径加载。绝对源路径只作为 provenance，不作为运行依赖。

## 最低验证

导入完成后必须编译模型，并记录或断言：

- `nq`、`nv`、关节数、actuator 数、equality 数和总质量；
- body/joint/site/actuator 名称唯一，策略关节与预期一致；
- 关节类型、轴向、限位、传动关系、effort/velocity limit 与设计输入一致；
- 惯量和质量有效，左右镜像残差有报告；“左右一致”不能替代 CAD 参数真实性验证；
- visual 与 collision 分组正确，碰撞体不依赖视觉 mesh 的偶然顺序；
- 默认 keyframe、base 高度、足底/手部接触 site 和 IMU 坐标正确；
- 传感器、接触 body 顺序和奖励/观测消费顺序一致。

改变节点顺序、零位、轴向或闭链关系时，必须增加针对性回归测试。优先按名称比较变换和
keyframe，而不是按数组下标比较。仓库中的 `build_v034_train_xml.py`、
`build_v035_train_xml.py`、`test_pikachu_v034_xml_order.py` 和
`test_pikachu_v035_default_pose.py` 是当前参考实现。

## 接入任务前门槛

1. 构建脚本从快照重建成功；
2. MuJoCo 编译和静态审计通过；
3. `--agent zero` 下姿态、接触、关节方向和 reset 正常；
4. `--agent random` 下动作维度、限幅、奖励和终止有合理响应；
5. 资产常量、任务默认位姿和 MJCF keyframe 相互一致；
6. manifest、几何报告和相关测试随资产一同提交。

未通过以上门槛的模型可以作为 research artifact 保存，但不得标记为可训练主路径或部署模型。
