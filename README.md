# Kairos

个人工作与生活助手，也是一个从零学习 agent 开发的项目。

Kairos 是希腊语里“恰当的时机”。这个助手和普通聊天机器人最大的区别，是会在对的时候主动找你：早报、到点提醒、每周回顾。

核心部分（agent 循环、模型适配层、记忆）都是手写的，不依赖 agent 框架。学习计划见 [本地 14 周计划](docs/个人%20Agent%20助手%2014%20周计划.html)，实验记录见 [docs/learning-log.md](docs/learning-log.md)。

源码参考范围见 [本地项目参考清单](references/SOURCE-MANIFEST.md)。用户无特殊说明时，只参考该清单内的本地项目，不额外补充其他项目。

当前项目已完成阶段 0 的多轮对话和 token/费用统计，以及阶段 1 的 s1-1 至 s1-9。主程序现在通过 `kairos.runtime.AgentRuntime` 使用 Agent loop；`cli.py` 负责终端输入、命令和展示，`terminal_renderer.py` 负责 Rich 渲染。s1-10 的双模型对比和阶段 2 仍待完成。接手时先读 [HANDOFF.md](HANDOFF.md) 和 [CLAUDE.md](CLAUDE.md)。

## 当前架构

主程序只有一个对话入口，组件由统一运行时组装：

```text
kairos 主入口
  -> cli.run()                  CLI：输入、斜杠命令、展示
  -> AgentRuntime.respond()    会话、模型、工具、数据库和日志的协调
  -> run_agent_turn()           模型请求、工具执行和跨轮循环
  -> providers/                 OpenAI 兼容或 Anthropic Messages 适配器
  -> todos.py + SQLite          待办工具和持久化数据
```

`src/kairos/runtime.py` 是应用的组装和协调边界：它根据 `Settings` 创建模型适配器，初始化 SQLite 和待办工具注册表，保存当前会话的 messages，并把一次用户输入交给 Agent loop。`src/kairos/cli.py` 只负责终端输入和命令，`src/kairos/terminal_renderer.py` 负责把结构化结果渲染为终端内容。

协议实现位于 `src/kairos/providers/`：`OpenAICompatibleModel`、`AnthropicMessagesModel` 和共享的响应转换逻辑分别放在独立模块中。主程序只通过 `runtime.create_runtime()` 创建运行时，不再保留一次性 Agent 包装入口。

## 流式对话与工具调用

主 CLI 默认使用流式输出，OpenAI 兼容协议和 Anthropic Messages 均支持：

1. 正文片段收到就展示，同时累积完整模型响应；工具参数单独收集，不作为正文打印。
2. 本次模型响应正常结束且全部工具参数完整后，按返回顺序串行执行整批工具。
3. 把整批工具结果写入历史并交回模型，继续流式接收后续回答；没有工具调用时结束 Agent 回合。

终端使用 Rich 活动区域渲染 Markdown 和 Mermaid 代码面板；长回答生成时显示末尾，响应结束后保留全文。重定向到文件或管道时逐段输出原始正文，不写入终端重绘控制码。整轮结束只补充用量统计，不重复打印答案。思考内容保留在模型历史中，不混入正文；Anthropic 内容块顺序和思考签名保留用于后续请求。

代码调用 `runtime.respond(prompt, on_event=handler)` 可接收 `text_delta`（`text`、`request_number`）及模型请求、响应、错误和工具结果事件。未传 `on_event` 时沿用完整响应接口；注入的模型只有 `complete()` 时会逐次展示完整模型响应。流式读取失败不会自动重跑整个回合。

Ctrl+C 或断流时，未完成模型响应不写入历史；已执行工具及其结果保留，工具副作用不会自动撤销。工具批次中断时给剩余调用补齐未执行或结果未确认标记。JSONL 日志和最终结果仍记录完整模型响应与工具结果，不逐片段重复保存正文。

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

模型协议通过 `.env` 的 `KAIROS_PROVIDER` 选择：

```dotenv
KAIROS_PROVIDER=openai       # 当前示例是 DeepSeek 的 OpenAI 兼容接口
KAIROS_MODEL=deepseek-flash
KAIROS_API_KEY=...
```

`KAIROS_MODEL` 和 `GATEWAY_BASE_URL` 是必填配置，没有代码默认值；启动时缺少任一项会直接提示配置无效。

切换到 Anthropic Messages 时改成：

```dotenv
KAIROS_PROVIDER=anthropic
KAIROS_MODEL=deepseek-flash       # 网关支持 Anthropic 协议时保留网关模型别名
GATEWAY_BASE_URL=https://gateway.example/v1
# 可选：Anthropic SDK 的网关根地址；不填时自动去掉上面地址末尾的 /v1
ANTHROPIC_BASE_URL=https://gateway.example
```

当前 Anthropic 适配器使用 Python `anthropic` SDK；选择该协议前需要在运行环境安装它。SDK 的请求路径是 `/v1/messages`，因此 Anthropic SDK 使用的网关根地址不要重复包含 `/v1`。代码中的统一字段是 `Settings.gateway_base_url`，环境变量使用 `GATEWAY_BASE_URL`。

`KAIROS_API_KEY` 是通用的网关项目 Key；协议切换只改变请求格式和 Base URL，不需要更换 Key。

待办数据保存在本地 `data/kairos.sqlite3`；主程序需要 `.env` 中所选协议对应的 API key。运行时的 `log_path` 可以由代码调用 `create_runtime()` 时显式传入。

## 当前验证基线

离线测试覆盖主 CLI、统一运行时、Agent loop、两种协议的流式适配和终端渲染。使用真实 SDK 加模拟 SSE 验证正文提前显示、整批工具等待与串行执行、工具历史回传、用量汇总、截断拒绝和中断恢复；没有请求真实网关。

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
