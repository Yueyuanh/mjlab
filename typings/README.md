# mjlab/typings 目录说明

本目录存放 **MuJoCo 的 Python 类型存根（`.pyi`）**，用于类型检查器（pyright / ty）对 MuJoCo C++ 绑定做静态类型分析时抑制误报。

## 一、为什么需要

`mujoco` 是 C++ 库的 Python 绑定，类型信息不全。没有存根时，pyright/ty 对 `mujoco` 相关代码报大量 unknown/Any 错误。这里提交一份**平台无关、可复现**的存根，供类型检查使用。

## 二、目录结构

```
typings/
├── generate_mujoco_stubs.sh   # 存根生成脚本（pybind11-stubgen）
├── postprocess_stubs.py       # 存根后处理（去平台相关内容）
└── mujoco/                    # 生成的 MuJoCo 类型存根
    ├── __init__.pyi           # mujoco 包总入口
    ├── _callbacks.pyi         # 回调接口
    ├── _constants.pyi         # 常量
    ├── _enums.pyi             # 枚举类型（mjt* / mj* 系列）
    ├── _errors.pyi            # 异常类型
    ├── _functions.pyi         # 顶层函数（mj_step、mj_data 等）
    ├── _render.pyi            # 渲染相关
    ├── _specs.pyi             # MjSpec API（mjlab 大量使用）
    ├── _structs.pyi           # 结构体（mjModel、mjData、mjOption 等）
    └── viewer.pyi             # mujoco.viewer 存根
```

## 三、生成流程

### 1. `generate_mujoco_stubs.sh`

用 **pybind11-stubgen**（pin 在 `2.5.5`，保证输出可复现）对已安装的 `mujoco` 生成存根：

```bash
# 在 mjlab 项目根目录
bash typings/generate_mujoco_stubs.sh
```

要点：
- `uv run --no-sync --with pybind11-stubgen==2.5.5` 在已有环境里跑，不动依赖（`cpu/cu128` 冲突 extras 会让普通 `uv run` 重新 sync）。
- 生成 `mujoco` 与 `mujoco.viewer` 两份存根；后者拉入 GLFW，无头机器导入失败时跳过（已提交的 viewer stub 保留）。
- 输出经 `postprocess_stubs.py` 处理后再落盘。

### 2. `postprocess_stubs.py`

把 pybind11-stubgen 的输出变成**平台无关、可复现**：

| 处理 | 说明 |
|------|------|
| 删除 OpenGL 后端 stub | 移除 `cgl/egl/glfw/glx/osmesa` 及 `gl_context.pyi`、`renderer.pyi`、`rendering`——不同平台/MUJOCO_GL 下只存在一个，mjlab 不对渲染后端做类型检查 |
| 删除 `__init__.pyi` 中的后端导入 | 清理 `from mujoco.cgl import GLContext` 等及 `__all__` 对应名字 |
| 剥离绝对路径与内存地址 | 默认值/`# value = ...` 注释里的安装路径和对象 repr 随机器变化，全部去掉 |

处理后剩下的就是一套**确定性**的物理绑定存根，每个平台完全一致。因此 CI 可以重新生成并 diff，若与已安装 mujoco 版本漂移则失败。

## 四、与构建/CI 的关系

- 存根是**提交进仓库**的（不是生成后丢弃），保证类型检查在任何机器上一致。
- CI 负责重新生成并对比：漂移即报错，防止提交的存根与依赖版本脱节。
- pyright 配置中 `exclude` 不含 `typings`，即存根目录参与类型解析（见根目录 [pyproject.toml](../pyproject.toml)）。

## 五、何时需要重新生成

当 `mujoco` 依赖版本升级、或新增代码用到 MuJoCo 新 API 而类型检查报错时：

```bash
uv sync            # 先同步新依赖
bash typings/generate_mujoco_stubs.sh
```

如果生成的存根与提交版有差异（正常情况），把新的 `typings/mujoco/` 一并提交。
