"""Kairos 的交互式命令行界面。

CLI 负责读取输入、处理斜杠命令和展示结果；模型、工具、数据库、消息历史和
Agent 循环由 ``kairos.runtime.AgentRuntime`` 统一管理。
"""

from pathlib import Path

import openai
from pydantic import ValidationError
from rich.console import Console

from .config import Settings
from .runtime import AgentRuntime, create_runtime
from .terminal_renderer import TerminalRenderer

__all__ = ["run"]

HELP = """\
[bold]命令[/]
  /history        查看当前的 messages 列表（下一次请求会把它整个发出去）
  /system 文本     替换 system 提示词（会清空对话，方便做对比实验）
  /think          开关思考模式
  /file 路径       把一个文本文件的内容作为你的消息发出去
  /reset          清空对话，保留 system 提示词
  /exit           退出（也可以按 Ctrl+D）
生成过程中按 Ctrl+C 可以中断这一轮。"""


def run(
    *,
    console: Console | None = None,
    settings: Settings | None = None,
    runtime: AgentRuntime | None = None,
) -> None:
    """启动交互式 CLI；可注入运行时以便离线测试和复用。"""

    console = console or Console()
    renderer = TerminalRenderer(console)
    owns_runtime = runtime is None
    if runtime is None:
        try:
            settings = settings or Settings()
        except ValidationError:
            renderer.error("[red]配置无效。[/]把 .env.example 复制成 .env，并检查模型配置后再运行。")
            return

        if not settings.kairos_api_key:
            renderer.error("[red]没有找到 KAIROS_API_KEY。[/]把 .env.example 复制成 .env，填入网关项目 Key 再运行。")
            return
        runtime = create_runtime(settings=settings)

    assert runtime is not None
    model = runtime.model_name
    think = False
    total_cost = 0.0

    renderer.status(f"[bold]Kairos[/] · 模型 {model} · 输入 /help 查看命令")

    try:
        while True:
            try:
                user_input = console.input("\n[bold cyan]你 › [/]").strip()
            except (EOFError, KeyboardInterrupt):
                renderer.status("")
                break
            if not user_input:
                continue

            # ---- 命令 ----
            if user_input.startswith("/"):
                cmd, _, arg = user_input.partition(" ")
                if cmd == "/exit":
                    break
                elif cmd == "/help":
                    renderer.status(HELP)
                elif cmd == "/history":
                    renderer.history(runtime.history())
                elif cmd == "/reset":
                    runtime.reset()
                    renderer.status("[dim]对话已清空。[/]")
                elif cmd == "/system":
                    if not arg:
                        renderer.status(f"[dim]当前 system 提示词：{runtime.system_prompt}[/]")
                        continue
                    runtime.reset(system_prompt=arg)
                    renderer.status("[dim]system 提示词已替换，对话已清空。[/]")
                elif cmd == "/think":
                    think = not think
                    runtime.set_thinking(think)
                    renderer.status(f"[dim]思考模式：{'开' if think else '关'}[/]")
                elif cmd == "/file":
                    path = Path(arg).expanduser()
                    try:
                        user_input = path.read_text(encoding="utf-8")
                    except OSError as e:
                        renderer.error(f"[red]读不了这个文件：{e}[/]")
                        continue
                    renderer.status(f"[dim]已读取 {path.name}，{len(user_input):,} 个字符。[/]")
                else:
                    renderer.error(f"[red]不认识的命令 {cmd}[/]，输入 /help 查看。")
                if cmd != "/file":
                    continue

            # ---- 一轮 Agent 对话 ----
            renderer.status("[bold green]Kairos › [/]")
            try:
                with renderer.stream_output():
                    result = runtime.respond(user_input, on_event=renderer.handle_event)
            except KeyboardInterrupt:
                renderer.status("\n[yellow]已中断。未完成的模型回答不计入历史，已执行的工具不会撤销。[/]")
                continue
            except openai.AuthenticationError:
                renderer.error("\n[red]API key 无效。[/]检查 .env 里的 KAIROS_API_KEY。")
                continue
            except openai.APIStatusError as e:
                renderer.error(f"\n[red]请求失败（HTTP {e.status_code}）：[/]{e.message}")
                continue
            except openai.APIConnectionError:
                renderer.error("\n[red]连不上服务器。[/]检查网络后重试。")
                continue
            except Exception as e:
                renderer.error(f"\n[red]本轮执行失败：{type(e).__name__}: {e}[/]")
                continue

            total_cost += renderer.render_turn(result, model=model, total_cost=total_cost, content_streamed=True)
    finally:
        if owns_runtime:
            runtime.close()
