"""Kairos 的交互式命令行界面。

CLI 负责读取输入、处理斜杠命令和展示结果；模型、工具、数据库、消息历史和
Agent 循环由 ``kairos.runtime.AgentRuntime`` 统一管理。
"""

import sqlite3
import sys
from pathlib import Path

import openai
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape

from .config import Settings
from .images import redact_images
from .runtime import AgentRuntime, create_runtime
from .terminal_input import InputDraft, TerminalInput, TerminalInputUnsupportedError
from .terminal_renderer import TerminalRenderer

__all__ = ["run"]

HELP = """\
[bold]命令[/]
  /new 标题        新建并切换会话，标题可省略，保留旧会话
  /sessions       列出已保存的会话
  /resume ID      恢复会话（支持唯一 ID 前缀）
  /rename 标题     修改当前会话标题
  /history        查看会话历史（图片显示摘要）
  /history --ids  查看会话、回合、步骤、消息和工具调用的 ID 关联
  /system 文本     使用新 system 提示词开启会话，保留旧会话
  /think          开关思考模式（仅已接入的模型）
  /file 路径       把一个文本文件的内容作为你的消息发出去
  /reset          新建会话，保留 system 提示词和旧会话
  /exit           退出（也可以按 Ctrl+D）
图片：Ctrl+V 粘贴图片，或拖入图片文件；编辑问题后 Enter 一起发送。
在图片标签右侧按 Backspace 可以移除附件。
生成过程中按 Ctrl+C 可以中断这一轮。"""


def run(
    *,
    console: Console | None = None,
    settings: Settings | None = None,
    runtime: AgentRuntime | None = None,
    input_editor: TerminalInput | None = None,
    session_id: str | None = None,
    resume_latest: bool = False,
) -> None:
    """启动终端交互并连接会话命令与 Agent 运行时。

    作用:
        创建或选择启动会话，读取用户输入，通过 Runtime 执行模型和会话操作，
        再展示流式回答与错误；新建或切换成功时清理附件草稿。

    参数:
        console: 可选输出终端；省略时创建 Rich Console。
        settings: 创建 Runtime 时使用的配置；已有注入运行时时无需创建模型配置。
        runtime: 可选的已有运行时，便于复用和离线测试；其关闭权留给调用方。
        input_editor: 可选输入编辑器，read() 返回 InputDraft；退出时由本函数关闭。
        session_id: 启动时恢复的完整会话 ID 或唯一前缀，和 resume_latest 互斥。
        resume_latest: 为真时继续最近会话；没有可恢复会话时显示错误并退出。

    返回:
        无返回值（None）。正常退出、输入结束或已处理的启动错误均结束函数。

    异常:
        ValueError: 本函数在以下位置显式抛出，随后由内部错误分支捕获并展示:
            - session_id 与 resume_latest 同时指定;
            - 已注入 Runtime，要求继续最近会话，但会话列表为空;
            - /history 带有非空且不是 --ids 的参数;
            - /resume 未提供会话 ID;
            - /rename 未提供标题。

    行为说明:
        1. 校验启动参数，创建或恢复 Runtime，准备输入编辑器并显示当前会话。
        2. 将本地命令交给会话接口，将普通输入交给 respond()；根据保留状态恢复失败草稿。
        3. 退出时始终关闭输入编辑器，仅关闭本函数创建的 Runtime，保留外部注入实例。
    """

    console = console or Console()
    renderer = TerminalRenderer(console)
    owns_runtime = runtime is None
    restore_draft: InputDraft | None = None
    try:
        try:
            if session_id is not None and resume_latest:
                raise ValueError("--session 和 --continue 不能同时使用")
            if runtime is None:
                try:
                    settings = settings or Settings()
                except ValidationError:
                    renderer.error("[red]配置无效。[/]把 .env.example 复制到 ~/.kairos/.env，并检查模型配置后再运行。")
                    return

                if not settings.kairos_api_key:
                    renderer.error(
                        "[red]没有找到 KAIROS_API_KEY。[/]"
                        "把 .env.example 复制到 ~/.kairos/.env，填入网关项目 Key 再运行。"
                    )
                    return
                runtime = create_runtime(settings=settings, session_id=session_id, resume_latest=resume_latest)
            elif session_id is not None:
                runtime.resume_session(session_id)
            elif resume_latest:
                sessions = runtime.list_sessions()
                if not sessions:
                    raise ValueError("没有可以恢复的会话，请直接运行 kairos 新建会话")
                runtime.resume_session(sessions[0].session_id)
        except (ValueError, OSError, RuntimeError, sqlite3.Error) as error:
            renderer.error(f"[red]无法启动会话：{escape(str(redact_images(str(error))))}[/]")
            return

        assert runtime is not None
        if input_editor is None and sys.stdin.isatty() and getattr(console, "is_terminal", False):
            try:
                input_editor = TerminalInput(budget_check=runtime.check_image_budget)
            except TerminalInputUnsupportedError:
                renderer.status("[dim]当前终端使用文字输入模式；图片粘贴和拖入需要支持交互的终端。[/]")

        renderer.status(f"[bold]Kairos[/] · {_session_label(runtime)} · 输入 /help 查看命令")
        _show_recovery_notice(runtime, renderer)
        if getattr(runtime, "memory_extractor", None) is not None:
            renderer.status("[dim]全局记忆已启用；每轮后台提取的用量单独记录，不计入前台费用统计。[/]")

        while True:
            _show_memory_events(runtime, renderer)
            try:
                if input_editor is not None:
                    draft = input_editor.read(draft=restore_draft)
                    restore_draft = None
                else:
                    draft = InputDraft(text=console.input("\n[bold cyan]你 › [/]").strip())
            except (EOFError, KeyboardInterrupt):
                renderer.status("")
                break
            user_input = draft.text.strip()
            if not user_input and not draft.images:
                continue

            # ---- 命令 ----
            if user_input.startswith("/"):
                cmd, _, arg = user_input.partition(" ")
                arg = arg.strip()
                try:
                    if cmd == "/exit":
                        break
                    elif cmd == "/help":
                        renderer.status(HELP)
                    elif cmd == "/history":
                        if arg == "--ids":
                            console.print_json(data=runtime.session_details())
                        elif not arg:
                            renderer.history(runtime.history())
                        else:
                            raise ValueError("用法：/history 或 /history --ids")
                    elif cmd == "/sessions":
                        _show_sessions(runtime, renderer)
                    elif cmd in {"/new", "/reset", "/resume"} or (cmd == "/system" and arg):
                        if cmd == "/new":
                            runtime.new_session(title=arg or None)
                        elif cmd == "/resume":
                            if not arg:
                                raise ValueError("用法：/resume ID")
                            runtime.resume_session(arg)
                        else:
                            runtime.reset(system_prompt=arg if cmd == "/system" else None)
                        restore_draft = None
                        if input_editor is not None:
                            input_editor.clear()
                        renderer.status(f"[dim]已切换 · {_session_label(runtime)}[/]")
                        _show_recovery_notice(runtime, renderer)
                    elif cmd == "/rename":
                        if not arg:
                            raise ValueError("用法：/rename 标题")
                        runtime.rename_session(arg)
                        renderer.status(f"[dim]已重命名 · {_session_label(runtime)}[/]")
                    elif cmd == "/system":
                        renderer.status(f"[dim]当前 system 提示词：{escape(runtime.system_prompt)}[/]")
                    elif cmd == "/think":
                        runtime.set_thinking(not runtime.thinking_enabled)
                        renderer.status(f"[dim]思考模式：{'开' if runtime.thinking_enabled else '关'}[/]")
                    elif cmd == "/file":
                        path = Path(arg).expanduser()
                        try:
                            user_input = path.read_text(encoding="utf-8")
                        except (OSError, UnicodeError) as e:
                            renderer.error(f"[red]读不了这个文本文件：{escape(str(e))}[/]")
                            if input_editor is not None:
                                restore_draft = draft
                            continue
                        renderer.status(f"[dim]已读取 {escape(path.name)}，{len(user_input):,} 个字符。[/]")
                    else:
                        renderer.error(f"[red]不认识的命令 {escape(cmd)}[/]，输入 /help 查看。")
                except (ValueError, OSError, RuntimeError, sqlite3.Error) as error:
                    renderer.error(f"[red]命令执行失败：{escape(str(redact_images(str(error))))}[/]")
                    continue
                if cmd != "/file":
                    continue

            # ---- 一轮 Agent 对话 ----
            # /file 展开的正文可能以 /reset 等开头；重试恢复原草稿，不能把
            # 文件正文重新解释成 CLI 命令。图片仍使用已经固定的内存快照。
            submitted = draft
            renderer.status("[bold green]Kairos › [/]")
            try:
                with renderer.stream_output():
                    if draft.images:
                        result = runtime.respond(
                            user_input,
                            images=draft.images,
                            image_labels=draft.image_labels,
                            on_event=renderer.handle_event,
                        )
                    else:
                        result = runtime.respond(user_input, on_event=renderer.handle_event)
            except KeyboardInterrupt:
                renderer.status("\n[yellow]已中断。未完成的模型回答不计入历史，已执行的工具不会撤销。[/]")
            except openai.AuthenticationError:
                renderer.error("\n[red]API key 无效。[/]检查 ~/.kairos/.env 里的 KAIROS_API_KEY。")
            except openai.APIStatusError as e:
                renderer.error(f"\n[red]请求失败（HTTP {e.status_code}）：[/]{escape(str(redact_images(e.message)))}")
            except openai.APIConnectionError:
                renderer.error("\n[red]连不上服务器。[/]检查网络后重试。")
            except Exception as e:
                message = escape(str(redact_images(str(e))))
                renderer.error(f"\n[red]本轮执行失败：{type(e).__name__}: {message}[/]")
            else:
                renderer.render_turn(result, model=runtime.model_name, content_streamed=True)
                continue

            if input_editor is not None:
                if runtime.last_input_retained:
                    renderer.status("[dim]本轮历史已保留，请继续询问；不会重新提交同一条消息。[/]")
                else:
                    restore_draft = submitted
                    renderer.status("[dim]草稿已恢复，可编辑后重新发送。[/]")
    finally:
        if input_editor is not None:
            input_editor.close()
        if owns_runtime and runtime is not None:
            if getattr(runtime, "memory_pending_count", 0):
                renderer.status("[dim]正在完成后台记忆提取，完成后退出。[/]")
            runtime.close()
            _show_memory_events(runtime, renderer)


def _show_memory_events(runtime: AgentRuntime, renderer: TerminalRenderer) -> None:
    """在主线程展示后台记忆保存结果。

    作用:
        在下一次输入和正常退出时报告实际写入或失败，不从后台线程重绘终端。

    参数:
        runtime: 提供 drain_memory_events 的运行时；兼容没有该接口的外部测试替身。
        renderer: 当前终端的统一输出器。

    返回:
        None。

    行为说明:
        1. 无新增结果保持安静；展示核心更新、主题更新数量及冲突、部分完成和失败。
        2. 不展示完整记忆、认证配置或模型错误正文；审计保存在记忆目录的 JSONL 中。
    """
    drain = getattr(runtime, "drain_memory_events", None)
    if not callable(drain):
        return
    for event in drain():
        status = event.get("status")
        if status == "completed":
            core_count = event.get("core_additions", 0)
            topic_count = len(set(event.get("topic_ids", [])))
            changes = []
            if core_count:
                changes.append("核心记忆已更新")
            if topic_count:
                changes.append(f"已更新 {topic_count} 个主题")
            if changes:
                renderer.status(f"[dim]后台记忆：{'；'.join(changes)}。[/]")
        elif status == "conflict":
            renderer.status("[yellow]后台记忆发现冲突，未覆盖原记录。[/]")
        elif status == "partial":
            renderer.status("[yellow]后台记忆已保存部分内容，但本轮提取未完整完成。[/]")
        elif status == "failed":
            renderer.status("[yellow]后台记忆提取失败；详情见 ~/.kairos/memory/extraction.jsonl。[/]")


def _session_label(runtime: AgentRuntime) -> str:
    """生成当前会话的终端摘要文字。

    作用:
        为启动、切换和重命名提示提供即时信息，避免显示旧会话的模型或思考状态。

    参数:
        runtime: 已激活且未关闭的运行时，提供最新 session_info 和模型状态。

    返回:
        包含短会话 ID、标题、模型与思考开关的字符串（str），动态文本已转义 Rich 标记。

    行为说明:
        1. 从 Runtime 读取当前会话概要和模型状态。
        2. 将会话 ID 截为前 8 位并转义动态文本，返回用于展示的摘要，不修改会话。
    """

    info = runtime.session_info
    thinking = f"思考 {'开' if runtime.thinking_enabled else '关'}" if runtime.supports_thinking else "思考切换未接入"
    return (
        f"会话 {escape(info.session_id[:8])} · {escape(info.title or '新会话')}"
        f" · 模型 {escape(runtime.model_name)} · {thinking}"
    )


def _show_recovery_notice(runtime: AgentRuntime, renderer: TerminalRenderer) -> None:
    """提示恢复过程中修补的执行记录和思考设置。

    作用:
        在启动或切换后说明上次执行尚未完成，帮助用户核实工具的实际执行结果。

    参数:
        runtime: 已完成加载的运行时，recovered_turns 表示本次修复过的回合数。
            thinking_state_corrected 表示本次纠正了未接入模型的旧思考开启状态。
        renderer: 用于显示提示的终端渲染器。

    返回:
        无返回值（None）；没有修补记录时不输出提示。

    行为说明:
        1. 检查 recovered_turns 是否非零。
        2. 有修补记录时显示检查历史的提示；修复工作已由 Runtime 完成，此处仅展示。
        3. 旧会话的无效思考设置已纠正时，说明实际支持情况及保存结果。
    """

    if runtime.recovered_turns:
        renderer.status("[yellow]上次未完成的执行已标记中断；请先查看历史，工具不会自动重跑。[/]")
    if runtime.thinking_state_corrected:
        renderer.status("[yellow]当前模型的思考切换尚未接入，旧会话的开启状态已纠正为关闭。[/]")


def _show_sessions(runtime: AgentRuntime, renderer: TerminalRenderer) -> None:
    """展示最近会话列表并标记当前会话。

    作用:
        供 /sessions 命令查找恢复入口，通过 Runtime 读取列表，再输出短 ID、时间和标题。

    参数:
        runtime: 提供 list_sessions() 和当前 session_id 的运行时。
        renderer: 用于显示列表或空列表提示的终端渲染器。

    返回:
        无返回值（None）；列表为空时显示提示后返回。

    行为说明:
        1. 获取按最近活动排序的会话列表，处理空列表情况。
        2. 短 ID 默认取前 8 位，在本次列表内出现歧义时延长，并转义动态文本。
        3. 为当前会话加星号标记；真正恢复时仍由 Runtime 校验完整查询范围内的 ID。
    """

    sessions = runtime.list_sessions()
    if not sessions:
        renderer.status("[dim]还没有已保存的会话；发送第一条消息或设置标题后会出现在这里。[/]")
        return
    renderer.status("[bold]已保存的会话[/]（* 当前会话）")
    for info in sessions:
        length = 8
        while length < len(info.session_id) and any(
            other.session_id != info.session_id and other.session_id[:length] == info.session_id[:length]
            for other in sessions
        ):
            length += 1
        marker = "*" if info.session_id == runtime.session_id else " "
        renderer.status(
            f"{marker} {escape(info.session_id[:length])}  {escape(str(info.updated_at))}  "
            f"{escape(info.title or '新会话')}"
        )
