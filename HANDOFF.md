# Kairos 交接说明

## 项目目标

Kairos 是个人工作与生活助手，也是从零学习 agent 开发的实践项目。长期目标是让助手在合适的时间主动提供早报、提醒和回顾；当前先通过手写基础组件理解 agent 的执行链。

## 当前状态

- 项目按正式应用维护，当前仓库用于持续开发。
- 当前代码完成阶段 0：命令行多轮对话、流式输出、思考模式切换、消息历史查看、文件输入，以及 token/费用统计。
- `src/kairos/config.py` 从项目根目录 `.env` 和环境变量读取模型协议配置；统一网关地址字段是 `Settings.gateway_base_url`，环境变量使用 `GATEWAY_BASE_URL`；`src/kairos/pricing.py` 负责用量转换和费用估算。
- 阶段 1 已完成 s1-1 至 s1-9：SQLite 待办函数、工具 Schema、支持同一响应多个工具调用、跨轮继续调用、工具错误回传、最大工具响应轮数和 JSONL 观测日志的手写循环，以及把 OpenAI 兼容格式和 Anthropic Messages 格式转换到同一内部接口的模型适配层。主入口 `kairos -> cli.run() -> AgentRuntime.respond() -> run_agent_turn()` 已接通 Agent loop 和正文流式输出；可复用运行时位于 `src/kairos/runtime.py`，协议实现位于 `src/kairos/providers/`。
- 流式策略：正文即时展示，每次模型流正常结束、参数完整后才串行执行整批工具，再把结果交回模型。中断不保存残缺模型响应，但保留已执行的工具结果；不自动重跑整个回合。详细调用与展示规则见 README 的“流式对话与工具调用”。
- 阶段 1 的 s1-10（同一批请求在两家模型上运行并比较工具调用格式和行为）尚未完成；阶段 2 尚未开始。
- `src/kairos/observability.py` 提供本地 JSONL 日志；通过 `create_runtime(log_path=...)` 显式启用，默认不写日志。
- 图片第一版已接入：macOS Ctrl+V／拖入完整图片路径 → prompt_toolkit 附件草稿 → Enter 提交图文；`images.py` 校验、方向校正、按需缩放压缩并固定内存快照。两种协议共用运行时和现有工具循环，历史/事件/日志只展示图片摘要。使用方式、限额和终端兼容边界见 README“图片输入”。
- 待办工具已增加 `delete_todo(todo_id)`：按 ID 永久删除未完成或已完成的记录，返回被删除的完整记录；找不到 ID 时通过现有工具错误流程回传。模型提示词要求仅在用户明确要求删除时调用，未提供 ID 时先查询定位，目标不明确时询问。

## 正常运行流程

1. 安装 Python 3.13、`uv`。
2. 复制 `.env.example` 为 `.env`，设置 `KAIROS_PROVIDER`、`KAIROS_MODEL`、通用的 `KAIROS_API_KEY` 和 `GATEWAY_BASE_URL`；`KAIROS_MODEL` 与 `GATEWAY_BASE_URL` 没有代码默认值，选择 `anthropic` 时，在需要时再设置不含 `/v1` 的 `ANTHROPIC_BASE_URL`。
3. 在项目根目录运行 `uv run kairos`，输入 `/help` 查看命令。主程序由 `cli.py` 提供 CLI，由 `terminal_renderer.py` 负责 Rich 展示，由 `AgentRuntime` 管理会话、模型、工具和 SQLite。
4. 运行 `uv run pytest -q` 和 `uv run ruff check .` 做基础检查；流式离线测试不消耗 API 额度。
5. 使用 VS Code 的 `Kairos CLI` 配置进入主程序，在 `cli.py`、`terminal_renderer.py`、`runtime.py`、`agent_loop.py` 和 `todos.py` 的边界设置断点。

真实密钥和个人数据不入库；流式改造最初使用模拟数据验收。图片接入阶段追加的真实网关测试仅发送合成图片，并使用临时待办数据库。

## 关键决定

- 核心 agent 循环、模型适配层和记忆层手写，不在早期引入 LangChain、LiteLLM 等框架；框架只在阶段 8 做对比。
- 上游使用用户自己的本地 AI 网关；网关仅透传请求和响应，不做其他处理。调用链与职责说明见 [README 的“上游 AI 网关”](README.md#上游-ai-网关)。
- 当前默认模型为 `deepseek-flash`，通过 OpenAI 兼容格式调用；价格会变化，修改价格表前先核对 DeepSeek 官方定价页。
- 注释和用户可见文字使用中文，实验过程和结论写入 `docs/learning-log.md`。

## 已验证与未验证

已验证：2026-10-12 图片接入后，177 项离线测试通过，Ruff 检查、格式检查和差异检查通过；真实 `xterm-256color` PTY 完成 Ctrl+V、图片路径粘贴、两轮图文提交及退出，功能受限终端回退文字模式。完整实验记录见 `docs/learning-log.md` 的实验 11。

已验证：2026-10-12 增加删除待办后，离线测试为 109 passed，Ruff 检查、格式检查和差异检查通过。删除测试使用临时数据库，覆盖目标隔离、持久化、缺失 ID 和工具结果回传；未操作实际待办，也未请求真实模型验收删除指令。

已验证：2026-10-12 本次流式改造通过 `.venv/bin/pytest -q`（103 passed）、Ruff 检查和格式检查。

已验证：主链路测试覆盖消息顺序、同一响应多个工具调用、跨轮继续调用、工具错误回传、最大轮数和 SQLite 工具执行；流式测试覆盖两种协议分片、正文即时展示、整批工具等待、串行执行、usage 汇总、Anthropic 原始内容回放、流关闭、截断拒绝和中断后历史配对。真实 SDK 的 SSE 解码通过内存 HTTP 模拟验证。

已验证：2026-10-12 图片链路在现有网关的 OpenAI、Anthropic 两种协议下均完成合成图片随机代号识别、文字追问、创建临时待办，全程正文流式输出；图片请求与工具结果历史正常回放，JSONL 未包含图片 Base64。两个协议使用同一个 `deepseek-flash`，不等于两家模型对比。

未验证：两家模型的真实对比，以及用户实际终端的 Finder 手工拖放、Cmd+V 和完整使用体验。输入区有 pipe input 自动测试；原生剪贴板用隔离 pasteboard 测试，不影响用户当前剪贴板。

## 下一步

下一步完成阶段 1 的 s1-10 双模型对比，并体验实际终端的图片粘贴和拖放；阶段 2 的工具设计与安全策略在此之后开始。每完成一个独立阶段，更新 README、学习记录和本交接文件。

## 文档入口

- [README.md](README.md)：项目目标、快速开始和阶段进度。
- [CLAUDE.md](CLAUDE.md)：协作方式、项目约定和已知坑。
- [docs/learning-log.md](docs/learning-log.md)：阶段实验记录。
- [docs/个人 Agent 助手 14 周计划.html](docs/个人%20Agent%20助手%2014%20周计划.html)：学习计划原文。
