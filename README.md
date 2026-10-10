# Kairos

个人工作与生活助手，也是一个从零学习 agent 开发的项目。

Kairos 是希腊语里“恰当的时机”。这个助手和普通聊天机器人最大的区别，是会在对的时候主动找你：早报、到点提醒、每周回顾。

核心部分（agent 循环、模型适配层、记忆）都是手写的，不依赖 agent 框架。学习计划见 [本地 14 周计划](docs/个人%20Agent%20助手%2014%20周计划.html)，实验记录见 [docs/learning-log.md](docs/learning-log.md)。

当前项目已完成阶段 0 的多轮对话和 token/费用统计，以及阶段 1 的 s1-1 至 s1-9。主程序现在通过 `kairos.runtime.AgentRuntime` 使用 Agent loop；`chat.py` 只负责交互式输入、命令和展示。s1-10 的双模型对比和阶段 2 仍待完成。接手时先读 [HANDOFF.md](HANDOFF.md) 和 [CLAUDE.md](CLAUDE.md)。

## 当前架构

主程序只有一个对话入口，组件由统一运行时组装：

```text
kairos 主入口
  -> chat.run()                 CLI：输入、斜杠命令、展示
  -> AgentRuntime.respond()    会话、模型、工具、数据库和日志的协调
  -> run_agent_turn()           模型请求、工具执行和跨轮循环
  -> providers/                 OpenAI 兼容或 Anthropic Messages 适配器
  -> todos.py + SQLite          待办工具和持久化数据
```

`src/kairos/runtime.py` 是应用的组装和协调边界：它根据 `Settings` 创建模型适配器，初始化 SQLite 和待办工具注册表，保存当前会话的 messages，并把一次用户输入交给 Agent loop。`src/kairos/chat.py` 不再自己创建模型或维护 Agent 消息历史。

协议实现位于 `src/kairos/providers/`：`OpenAICompatibleModel`、`AnthropicMessagesModel` 和共享的响应转换逻辑分别放在独立模块中。旧的 `src/kairos/model.py` 仍保留兼容导出；`src/kairos/agent_app.py` 也保留 `run_cli`、`run_prompt` 等阶段 1 调用方的兼容包装。`scripts/debug_agent.py` 是一次性真实模型调试入口，复用 `create_runtime()`，不是另一套 Agent 实现。

## 快速开始

需要 [uv](https://docs.astral.sh/uv/)。

```bash
cp .env.example .env      # 然后在 .env 里选择 KAIROS_PROVIDER 并填入对应的 key
uv run kairos
```

## 命令

| 命令 | 作用 |
|---|---|
| `/history` | 查看当前的 messages 列表（下一次请求会把它整个发出去） |
| `/system 文本` | 替换 system 提示词，并清空对话 |
| `/think` | 开关思考模式 |
| `/file 路径` | 把一个文本文件的内容作为消息发出去 |
| `/reset` | 清空对话 |
| `/exit` | 退出 |

一次性真实模型调试可以单独运行，不影响主程序的聊天历史：

```bash
uv run python scripts/debug_agent.py "把学习 Agent 加入待办"
```

Agent 调试入口通过 `.env` 的 `KAIROS_PROVIDER` 选择协议：

```dotenv
KAIROS_PROVIDER=openai       # 当前示例是 DeepSeek 的 OpenAI 兼容接口
KAIROS_MODEL=deepseek-flash
KAIROS_API_KEY=...
```

切换到 Anthropic Messages 时改成：

```dotenv
KAIROS_PROVIDER=anthropic
KAIROS_MODEL=deepseek-flash       # 网关支持 Anthropic 协议时保留网关模型别名
GATEWAY_BASE_URL=https://gateway.example/v1
# 可选：Anthropic SDK 的网关根地址；不填时自动去掉上面地址末尾的 /v1
ANTHROPIC_BASE_URL=https://gateway.example
```

当前 Anthropic 适配器使用 Python `anthropic` SDK；选择该协议前需要在运行环境安装它。SDK 的请求路径是 `/v1/messages`，因此 Anthropic SDK 使用的网关根地址不要重复包含 `/v1`。`Settings.gateway_base_url` 是代码中的统一字段；`KAIROS_BASE_URL`、`GATEWAY_BASE_URL`、`OPENAI_BASE_URL` 和旧的 `DEEPSEEK_BASE_URL` 都可作为环境变量别名读取。

`KAIROS_API_KEY` 是通用的网关项目 Key；协议切换只改变请求格式和 Base URL，不需要更换 Key。

可以运行离线实验观察工具错误回传，不请求真实模型：

```bash
uv run python scripts/experiment_tool_errors.py
```

它会依次模拟非法 JSON、未知工具和工具函数异常，并打印异常类型、模型调用次数和停止时的 `messages`。

VSCode 的运行和调试配置名为 `Kairos Tool Error Experiment`，默认运行 `handler-error`。想换案例时，把 `.vscode/launch.json` 里的 `args` 改成 `invalid-json` 或 `unknown-tool`，再按 F5；可在 `agent_loop.py` 的 `_execute_tool` 内设置断点逐步查看 `tool_call`、`arguments` 和错误结果。

真实 Agent 也可以写本地 JSONL 观测日志：

```bash
uv run python scripts/debug_agent.py --log-path /tmp/kairos-agent.jsonl "请查询当前所有未完成待办"
```

待办数据保存在本地 `data/kairos.sqlite3`；该命令需要 `.env` 中所选协议对应的 API key。
命令行脚本位于 `scripts/`，正式的 Agent 编排代码位于 `src/kairos/runtime.py` 和 `src/kairos/agent_loop.py`。

## 当前验证基线

在当前重构分支上，离线测试为 `44 passed`，`uv run ruff check .` 通过。这个基线覆盖运行时接入主程序、连续会话、待办工具注册、跨工具轮次、用量汇总、配置别名和两个模型适配器的 Fake Client 路径；尚未证明真实网关请求成功。

## 进度

- [x] 阶段 0：多轮对话、流式输出、token 和费用统计
- [ ] 阶段 1：手写 agent 循环 + 待办工具（s1-1 至 s1-9 已完成；s1-10 双模型对比待完成）
- [ ] 阶段 2：工具设计与安全
- [ ] 阶段 3：记忆与上下文管理
- [ ] 阶段 4：主动性
- [ ] 阶段 5：接入手机
- [ ] 阶段 6：MCP
- [ ] 阶段 7：评估与观测
- [ ] 阶段 8：框架对比

## 开发

```bash
uv run pytest -q
uv run ruff check .
uv run ruff check --fix && uv run ruff format
```
