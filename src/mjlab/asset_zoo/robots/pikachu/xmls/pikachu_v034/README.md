# Pikachu V034 MJCF 顺序

`mjcf/pikachu_v034.xml` 和 `mjcf/pikachu_v034_train.xml` 的连杆、关节及
mesh 声明顺序与 V033 `pikachu_v033_train.xml` 一致：

1. `base_link`（训练模型另含 `floating_base`）。
2. 右腿：hip_pitch → hip_roll → hip_yaw → knee_up → knee_mid → knee → ankle。
3. 左腿：同上。
4. 右臂：arm_pitch → arm_roll → arm_yaw → elbow。
5. 左臂：同上。

这是树的遍历顺序。膝部仍为原有两条兄弟分支：`knee_up → knee_mid` 和
`knee → ankle`，没有改变父子连接。

保留 V034 自身的轴向、限位、质量、惯量、局部位置/姿态、碰撞几何、阻尼、
摩擦、闭环约束、传感器及电机参数。`stand` 的 qpos 按关节名重排，保持原姿态。
`scene.xml`、电机声明顺序及传感器声明顺序保持原样。

原始文件及 SHA-256 保存在 `provenance/pre_v033_order/`。
`source_manifest.json` 的 `sha256` 仍记录导入原件，`transformations` 记录
重排后的源 XML 哈希。重新生成训练模型时，`scripts/build_v034_train_xml.py`
也会应用此顺序。

注意：MuJoCo 的 body/joint/qpos/qvel 索引随顺序改变。原始 Biped/Crawl
基线任务采用模型遍历顺序，旧权重需要转换关节观测和动作通道后使用；
Biped-E1/Crawl-E3 已显式指定策略关节顺序，其 actor 关节接口保持一致。
所有任务的原始接触观测通道也会随 body 顺序改变，恢复旧 critic/训练状态前
需要同步转换。历史训练配置和模型未修改。

验证：

```sh
uv run pytest tests/test_pikachu_v034.py tests/test_pikachu_v034_xml_order.py tests/test_pikachu_v034_v033_framework.py -q
```
