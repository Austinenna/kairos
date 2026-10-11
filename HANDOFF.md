# Kairos 交接说明

## 项目目标

Kairos 是个人工作与生活助手，也是从零学习 agent 开发的实践项目。长期目标是让助手在合适的时间主动提供早报、提醒和回顾；当前先通过手写基础组件理解 agent 的执行链。

## 当前状态

- 项目按正式应用维护，当前仓库用于持续开发。
- 当前代码完成阶段 0：命令行多轮对话、流式输出、思考模式切换、消息历史查看、文件输入，以及 token/费用统计。
- `src/kairos/config.py` 从项目根目录 `.env` 和环境变量读取模型协议配置；统一网关地址字段是 `Settings.gateway_base_url`，环境变量使用 `GATEWAY_BASE_URL`；`src/kairos/pricing.py` 负责用量转换和费用估算。
- 阶段 1 已完成 s1-1 至 s1-9：SQLite 待办函数、工具 Schema、支持同一响应多个工具调用、跨轮继续调用、工具错误回传、最大工具响应轮数和 JSONL 观测日志的手写循环，以及把 OpenAI 兼容格式和 Anthropic Messages 格式转换到同一内部接口的模型适配层。重构后主入口 `kairos -> chat.run() -> AgentRuntime.respond() -> run_agent_turn()` 已接通 Agent loop；可复用运行时位于 `src/kairos/runtime.py`，协议实现位于 `src/kairos/providers/`。
- 阶段 1 的 s1-10（同一批请求在两家模型上运行并比较工具调用格式和行为）尚未完成；阶段 2 尚未开始。
- `src/kairos/observability.py` 提供本地 JSONL 日志；通过 `--log-path` 显式启用，默认不写日志。

## 正常运行流程

1. 安装 Python 3.13、`uv`。
2. 复制 `.env.example` 为 `.env`，设置 `KAIROS_PROVIDER`、通用的 `KAIROS_API_KEY` 和 `GATEWAY_BASE_URL`；选择 `anthropic` 时，在需要时再设置不含 `/v1` 的 `ANTHROPIC_BASE_URL`。
3. 在项目根目录运行 `uv run kairos`，输入 `/help` 查看命令。主程序由 `chat.py` 提供 CLI，由 `AgentRuntime` 管理会话、模型、工具和 SQLite。
4. 运行 `uv run pytest -q` 和 `uv run ruff check .` 做基础检查；当前主链路基线为 `18 passed` 且 Ruff 通过。
5. 使用 VS Code 的 `Kairos CLI` 配置进入主程序，在 `chat.py`、`runtime.py`、`agent_loop.py` 和 `todos.py` 的边界设置断点。

真实密钥和个人数据不入库；当前只验证了不需要真实 API 请求的定价测试，命令行对话和远程模型调用尚未做本次转正验收。

## 关键决定

- 核心 agent 循环、模型适配层和记忆层手写，不在早期引入 LangChain、LiteLLM 等框架；框架只在阶段 8 做对比。
- 当前默认模型为 `deepseek-flash`，通过 OpenAI 兼容格式调用；价格会变化，修改价格表前先核对 DeepSeek 官方定价页。
- 注释和用户可见文字使用中文，实验过程和结论写入 `docs/learning-log.md`。

## 已验证与未验证

已验证：当前重构分支的主链路测试和 Ruff 检查通过。

已验证：保留的主链路测试覆盖消息顺序、同一响应多个工具调用、工具执行后的跨轮继续调用、工具错误回传、最大轮数、主 CLI 接入 Runtime、SQLite 工具执行和 usage 汇总。真实 provider 请求仍未验收。

未验证：Anthropic 适配器的真实网关 API 调用、两家模型的真实对比，以及真实网络下的完整命令行交互。当前主程序已经通过 `AgentRuntime` 接入统一 Agent loop，主流程使用 runtime 的完整回合结果。Fake Model 已验证主链路的消息顺序、工具执行和跨轮行为。

## 下一步

下一步完成阶段 1 的 s1-10 双模型对比，并单独验收真实网关调用；阶段 2 的工具设计与安全策略在此之后开始。每完成一个独立阶段，更新 README、学习记录和本交接文件。

## 文档入口

- [README.md](README.md)：项目目标、快速开始和阶段进度。
- [CLAUDE.md](CLAUDE.md)：协作方式、项目约定和已知坑。
- [docs/learning-log.md](docs/learning-log.md)：阶段实验记录。
- [docs/个人 Agent 助手 14 周计划.html](docs/个人%20Agent%20助手%2014%20周计划.html)：学习计划原文。
