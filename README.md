# Kairos

个人工作与生活助手，也是一个从零学习 agent 开发的项目。

Kairos 是希腊语里“恰当的时机”。这个助手和普通聊天机器人最大的区别，是会在对的时候主动找你：早报、到点提醒、每周回顾。

核心部分（agent 循环、模型适配层、记忆）都是手写的，不依赖 agent 框架。学习计划见 [本地 14 周计划](docs/个人%20Agent%20助手%2014%20周计划.html)，实验记录见 [docs/learning-log.md](docs/learning-log.md)。

当前项目已完成阶段 0 的最小多轮对话、流式输出和 token/费用统计；阶段 1 已完成 s1-1 至 s1-9，s1-10 的双模型对比仍待完成。Agent 调试入口已经可以在 OpenAI 兼容协议和 Anthropic Messages 协议之间切换。阶段 2 尚未开始。接手时先读 [HANDOFF.md](HANDOFF.md) 和 [CLAUDE.md](CLAUDE.md)。

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

阶段 1 的单轮真实模型调试可以单独运行，不影响阶段 0 的聊天历史：

```bash
uv run python scripts/debug_agent.py "把学习 Agent 加入待办"
```

Agent 调试入口通过 `.env` 的 `KAIROS_PROVIDER` 选择协议：

```dotenv
KAIROS_PROVIDER=openai       # 当前示例是 DeepSeek 的 OpenAI 兼容接口
KAIROS_MODEL=deepseek-flash
DEEPSEEK_API_KEY=...
```

切换到 Anthropic Messages 时改成：

```dotenv
KAIROS_PROVIDER=anthropic
KAIROS_MODEL=<Anthropic 模型名>
ANTHROPIC_API_KEY=...
```

当前 Anthropic 适配器使用 Python `anthropic` SDK；选择该协议前需要在运行环境安装它。阶段 0 的 `uv run kairos` 仍是独立的流式聊天入口，暂时只走 `chat.py` 的 OpenAI 兼容路径。

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
命令行脚本位于 `scripts/`，正式的 Agent 编排代码位于 `src/kairos/agent_app.py`。

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
uv run pytest
uv run ruff check --fix && uv run ruff format
```
