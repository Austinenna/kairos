# Kairos

个人工作与生活助手，也是一个从零学习 agent 开发的项目。

Kairos 是希腊语里“恰当的时机”。这个助手和普通聊天机器人最大的区别，是会在对的时候主动找你：早报、到点提醒、每周回顾。

核心部分（agent 循环、模型适配层、记忆）都是手写的，不依赖 agent 框架。学习计划见 [本地 14 周计划](docs/个人%20Agent%20助手%2014%20周计划.html)，实验记录见 [docs/learning-log.md](docs/learning-log.md)。

当前项目已完成阶段 0 的最小多轮对话、流式输出和 token/费用统计；阶段 1 已补齐待办工具的多工具调用、多轮继续、工具错误回传、最大工具轮数和截断提示。JSONL 观测日志、多模型适配和工具安全策略仍待后续阶段实现。接手时先读 [HANDOFF.md](HANDOFF.md) 和 [CLAUDE.md](CLAUDE.md)。

## 快速开始

需要 [uv](https://docs.astral.sh/uv/)。

```bash
cp .env.example .env      # 然后在 .env 里填入 DEEPSEEK_API_KEY
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

阶段 1 的真实模型调试可以单独运行，不影响阶段 0 的聊天历史：

```bash
uv run python scripts/debug_agent.py "把学习 Agent 加入待办"
```

待办数据保存在本地 `data/kairos.sqlite3`；该命令需要 `.env` 中的 `DEEPSEEK_API_KEY`。
命令行脚本位于 `scripts/`，正式的 Agent 编排代码位于 `src/kairos/agent_app.py`。

## 进度

- [x] 阶段 0：多轮对话、流式输出、token 和费用统计
- [ ] 阶段 1：手写 agent 循环 + 待办工具（已完成 SQLite、Schema、真实模型调试链路、多工具、多轮、错误回传、轮数上限和截断提示；JSONL 日志与多模型适配待补）
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
