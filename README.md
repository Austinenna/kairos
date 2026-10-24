# Kairos

个人工作与生活助手，也是一个从零学习 agent 开发的项目。

Kairos 是希腊语里“恰当的时机”。这个助手和普通聊天机器人最大的区别，是会在对的时候主动找你：早报、到点提醒、每周回顾（主动性尚在计划中）。

agent 循环、模型适配层、记忆都是手写的，不依赖 agent 框架。实验记录在本地的 `docs/learning-log.md`。

## 快速开始

需要 Python 3.13 和 [uv](https://docs.astral.sh/uv/)。

```bash
mkdir -p ~/.kairos && chmod 700 ~/.kairos
cp .env.example ~/.kairos/.env   # 填写 KAIROS_API_KEY、GATEWAY_BASE_URL、KAIROS_MODEL
chmod 600 ~/.kairos/.env
uv run kairos
```

```bash
uv run kairos                      # 新会话，同时启动终端和微信
uv run kairos --continue           # 继续最近的会话
uv run kairos --session ID         # 恢复指定会话（ID 可以只写开头几位）
uv run kairos --channels terminal  # 只启动终端（或 --channels weixin 只启动微信）
```

## 使用

### 终端

| 命令 | 作用 |
|---|---|
| `/new` | 新建会话 |
| `/sessions` | 列出最近的会话 |
| `/resume ID` | 恢复会话 |
| `/rename 标题` | 重命名当前会话 |
| `/think` | 切换思考模式（仅支持的模型） |
| `/history` | 查看当前会话的消息 |
| `/file 路径` | 把文本文件的内容作为消息发送 |
| `/help`、`/exit` | 帮助、退出（Ctrl+D 或连按两次 Ctrl+C 也可以退出） |

- **Enter** 发送，**Alt+Enter** 换行。生成回答时按 **Esc** 中断，已显示的部分会保留在历史里。
- 输入框里按 **Ctrl+C** 清空输入，2 秒内再按一次退出；生成中按 Ctrl+C 也会中断，紧接着再按一次退出。
- 输入框下方的状态行显示：模型、上下文占用（最近一次请求的输入量 / 窗口大小）、本会话累计用量和估算费用。

### 图片

- 截图后在输入框按 **Ctrl+V**，或者把图片文件拖进终端。
- 待发送的图片显示在输入框下方；输入框为空时按 **Backspace** 删除最后一张。
- 图片会按 EXIF 校正方向、缩到最长边 2000 像素，随消息保存在会话里，恢复会话后仍可追问。

### 微信

```bash
uv run --with 'qrcode[pil]' python -m kairos.wechat   # 首次扫码登录
```

登录后 `uv run kairos` 会同时启动微信入口。Bot 只回复扫码登录的本人，支持文字和图片。发来的图片会自动存档：模型看图后给出分类、标题和描述，原图存到 `~/.kairos/files/<分类>/`，并回复存放位置。发图后接着说的那句话会带上这些图（最后一张图之后 15 分钟内），可以直接问图里的内容；对话里较早的图只保留描述，模型需要时用 `view_image` 取回原图。之后在终端或微信里问“上次那张停车票存哪了”，模型会用 `search_images` 找出文件路径；微信里可以发 `/new` 开始新会话、`/help` 查看帮助。终端和微信可以同时使用，两边共用会话、记忆和待办。

### 记忆

记忆是 `~/.kairos/memories/` 下的普通 Markdown 文件，可以直接查看和修改：

- `CORE.md`：核心记忆，每次请求都会带上。用户说“记住……”时由模型写入，称呼、通用偏好等也会被后台自动写入。
- `topics/标题.md`：其他长期信息，一个主题一个文件。模型需要时按标题读取。

每轮对话结束后，后台会把最近 3 轮对话交给模型判断有没有值得记住的内容，写入后在下一次输入前提示。

### 待办和历史

- 待办：直接说“明天下午三点提醒我交房租”“列出待办”“删除买牛奶那条”。待办模块在 `src/kairos_todo/`，只依赖标准库，可以通过 `KAIROS_TOOL_FACTORIES` 关掉。
- 历史：可以问“之前我们讨论过……吗”“今天微信里聊了什么”，模型会搜索、按日期列出并阅读以前的会话。终端和微信的会话记在同一份历史里，`/sessions` 会标出每个会话来自哪个入口。

## 架构

```text
uv run kairos ─▶ terminal/app.py ─▶ open_services()（services.py，全部会话共用，只创建一次）
                      │                ├─ 数据库与会话存储   storage/
                      │                ├─ 工具集合           tools/ + kairos_todo
                      │                ├─ 记忆存储           memory/store.py
                      │                └─ 后台记忆提取       memory/extractor.py
                      │
                      ├─ 终端（主线程）──┐
                      └─ 微信（后台线程）┴─▶ Conversation（conversation.py，每个会话一份）
                                              respond()：保存用户消息 → run_turn → 提交记忆提取
                                                   │
                                                   ▼
                                     agent/loop.py  run_turn：请求模型 ⇄ 执行工具
                                     agent/context.py  请求视图：提示词 + 规则 + 历史 + 记忆和时间
                                     models/  OpenAI 兼容 / Anthropic 适配器、模型能力和价格表
```

| 目录或文件 | 作用 |
|---|---|
| `agent/loop.py` | Agent 循环：请求模型、串行执行工具、把结果交回模型，直到模型不再调用工具 |
| `agent/context.py` | 每次请求前构建请求视图；会变的内容放在末尾，让前缀缓存能命中 |
| `models/` | `base.py` 统一的 `ModelResponse`；两个协议适配器；`catalog.py` 模型能力和价格 |
| `conversation.py` | 一个会话：历史、模型、`respond()` |
| `services.py` | 共享依赖的组装；`lock` 让终端和微信轮流使用数据库 |
| `storage/sessions.py` | 两张表：`sessions`、`messages`（完整消息 JSON + 用量和费用） |
| `memory/` | 记忆文件读写、后台提取 |
| `tools/` | 工具注册表、记忆工具、历史会话工具、图片搜索工具 |
| `terminal/` | 主循环和命令、输入框（图片）、Markdown 流式渲染、读取 macOS 剪贴板 |
| `wechat/` | iLink 协议、微信收发线程 |
| `images.py` | 图片处理 |
| `archive.py` | 图片归档：存原图、`images` 表；`ImageClassifier` 在后台线程用自己的数据库连接让模型分类，不占用对话 |
| `image_refs.py` | 请求前把历史里的图片引用展开：最近一批给原图，更早的给描述 |

模型调用链：`Kairos → 本地 AI 网关 → 模型服务`。网关只透传请求，协议适配、Agent 循环和工具执行都在 Kairos 里。默认模型是 DeepSeek（`deepseek-flash`），走 OpenAI 兼容协议；设置 `KAIROS_PROVIDER=anthropic` 改用 Anthropic Messages 协议。

## 数据

全部放在当前用户的 `~/.kairos/`，不进 Git：

| 路径 | 内容 |
|---|---|
| `.env` | 配置和 API Key |
| `kairos.db` | 会话、消息、用量、待办，以及归档图片的信息（`images` 表） |
| `memories/` | 记忆 |
| `files/` | 归档的图片，按分类分文件夹；分类失败的留在 `inbox/` |
| `channels/weixin/` | 微信凭据 `account.json`、收发状态 `state.json` |

## 开发

```bash
uv run pytest                                      # 测试
uv run ruff check --fix && uv run ruff format      # 检查和格式化
```

开发原则见 `CLAUDE.md`：先写能用的主路径，只处理确定会发生的情况，遇到问题再补。

## 进度

- [x] 阶段 0：多轮对话、流式输出、token 和费用统计
- [ ] 阶段 1：手写 agent 循环 + 待办工具（s1-10 双模型对比待完成）
- [ ] 阶段 2：工具设计与安全
- [ ] 阶段 3：记忆与上下文管理（全局记忆已完成；长对话管理待做）
- [ ] 阶段 4：主动性
- [ ] 阶段 5：接入手机（微信文字私聊已接入，真实手机联调待做）
- [ ] 阶段 6：MCP
- [ ] 阶段 7：评估与观测
- [ ] 阶段 8：框架对比

2026-10-23 完成精简重构：源码从约 13,900 行减到约 2,200 行（不含待办模块），测试从约 18,100 行减到约 700 行。
