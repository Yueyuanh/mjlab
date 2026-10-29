# 将 spec 作为不同 Agent 的默认指令

## 设计

Agent 默认加载应采用两层结构：

```text
Agent 原生入口 -> spec/agent-defaults.md -> 与当前任务相关的主题规范
```

原生入口只负责引导和平台特有约定，避免把全部规范复制到多个文件。这样新开会话无需用户再次
要求“请阅读 spec”，同时又不会把无关主题全部塞进上下文。

本仓库已接入：

| Agent | 自动发现文件 | 当前状态 |
| --- | --- | --- |
| OpenAI Codex 及兼容工具 | 工作区与 `mjlab/` 的 `AGENTS.md` | 已接入 |
| Claude Code | 工作区与 `mjlab/` 的 `CLAUDE.md` | 已接入 |

两者都应在开始仓库工作时读取 `spec/agent-defaults.md`，再按其中路由读取主题文件。根文件中
已有的分支、同步、代码检查规则继续生效。

Codex 在会话启动时只从项目根沿路径读取到当前工作目录，并且每层最多读取一个指令文件。
因此本工作区在 `pikachu_playground/AGENTS.md` 保留跨目录路由，在 `mjlab/AGENTS.md` 保留
仓库级规则；从任一层启动都不需要用户再次提醒。此行为依据
[OpenAI 官方 AGENTS.md 文档](https://learn.chatgpt.com/docs/agent-configuration/agents-md)。

## 新增 Agent 的方法

为其他工具接入时，在它自动发现的仓库级文件中加入等价的短引导。例如：

| 工具 | 常见入口 |
| --- | --- |
| Cursor | `.cursor/rules/mjlab-spec.mdc`（设为 always apply） |
| GitHub Copilot | `.github/copilot-instructions.md` |
| Gemini CLI | `GEMINI.md` |
| 其他 Agent | 该工具明确支持的 repository instructions 文件 |

引导内容应表达：

```text
Before working in this repository, read spec/agent-defaults.md completely.
Then read every topic spec routed by the current task. These specs are mandatory
for new mjlab training work; preserve user changes and document any exception.
```

不要仅依赖 shell alias、个人全局 prompt 或聊天记忆，因为其他用户、CI Agent 和新会话无法继承。
不确定工具是否支持 Markdown import 时，使用普通文字指令，不依赖 `@file` 等私有语法。

## 验证接入

接入新 Agent 后，用一个只读测试会话询问：

1. 仓库训练规范入口在哪里；
2. 模型转换任务需要读取哪份主题规范；
3. 如何判定训练完成；
4. 是否可以用最高编号 checkpoint 自动当最终模型。

正确回答应指向本目录，并明确 `config/run.json` 的 `completed` 状态、最终 checkpoint 路径和
SHA-256。若 Agent 没有自动读取
原生入口，应修正工具配置，而不是要求每个任务提示词重复粘贴规范。

## 维护注意

- 强规则只在 `spec/` 修改，Agent 入口不复制正文；
- 新增主题后同步 `spec/README.md` 和 `agent-defaults.md` 路由；
- Agent 入口过长时，逐步把流程内容迁回 `spec/`，但保留平台特有的安全和代码规范；
- 可后续增加小型 CI 检查，确认所有已支持入口仍引用 `spec/agent-defaults.md`。
