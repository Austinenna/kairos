# Kairos 交接说明

## 项目目标

Kairos 是个人工作与生活助手，也是从零学习 agent 开发的实践项目。长期目标是让助手在合适的时间主动提供早报、提醒和回顾；当前先通过手写基础组件理解 agent 的执行链。

## 当前状态

- 项目按正式应用维护，当前仓库用于持续开发。
- 当前代码完成阶段 0：命令行多轮对话、流式输出、思考模式切换、消息历史查看、文件输入，以及 token/费用统计。
- `src/kairos/config.py` 从项目根目录 `.env` 和环境变量读取 DeepSeek 配置；`src/kairos/chat.py` 维护消息列表并调用 OpenAI 兼容接口；`src/kairos/pricing.py` 负责用量转换和费用估算。
- `tests/test_pricing.py` 覆盖高峰时段、缓存输入计费和未知模型价格。
- 阶段 1 已完成最小闭环：SQLite 待办函数、工具 Schema、一次最多执行一个工具的手写循环，以及 `DeepSeekModel` 和 `python scripts/debug_agent.py` 真实调试入口。可复用编排位于 `src/kairos/agent_app.py`，调试脚本单独放在 `scripts/`。当前仍不支持一次请求中的多个工具调用、工具执行后的继续调用和完整错误回传。

## 正常运行流程

1. 安装 Python 3.13、`uv`。
2. 复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY`；可按需调整 `DEEPSEEK_BASE_URL` 和 `KAIROS_MODEL`。
3. 在项目根目录运行 `uv run kairos`，输入 `/help` 查看命令。
4. 运行 `uv run pytest` 和 `uv run ruff check .` 做基础检查。

真实密钥和个人数据不入库；当前只验证了不需要真实 API 请求的定价测试，命令行对话和远程模型调用尚未做本次转正验收。

## 关键决定

- 核心 agent 循环、模型适配层和记忆层手写，不在早期引入 LangChain、LiteLLM 等框架；框架只在阶段 8 做对比。
- 当前默认模型为 `deepseek-flash`，通过 OpenAI 兼容格式调用；价格会变化，修改价格表前先核对 DeepSeek 官方定价页。
- 注释和用户可见文字使用中文，实验过程和结论写入 `docs/learning-log.md`。

## 已验证与未验证

已验证：迁移前后项目文件校验通过；Python 环境中的 3 项定价测试通过。

已验证：Fake Model 覆盖消息顺序和工具执行；Fake SDK Client 覆盖 DeepSeek 响应归一化，并覆盖真实适配器接入 SQLite 工具循环的组合路径。

未验证：`debug_agent` 的真实 API 调用、流式显示、思考模式、文件输入和完整命令行交互；这些需要配置真实密钥和网络后再单独验收。

## 下一步

下一步从阶段 1 的边界补强开始：先在真实调试入口中观察 API 错误、工具参数错误和多工具响应，再决定是否扩展最大轮数、错误回传和工具安全策略。每完成一个独立阶段，更新 README、学习记录和本交接文件。

## 文档入口

- [README.md](README.md)：项目目标、快速开始和阶段进度。
- [CLAUDE.md](CLAUDE.md)：协作方式、项目约定和已知坑。
- [docs/learning-log.md](docs/learning-log.md)：阶段实验记录。
- [docs/个人 Agent 助手 14 周计划.html](docs/个人%20Agent%20助手%2014%20周计划.html)：学习计划原文。
