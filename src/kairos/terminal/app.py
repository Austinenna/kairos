"""终端主循环：读输入 → 执行命令或发给 Conversation → 显示结果。

运行：uv run kairos [--continue | --session ID]
"""

import argparse
from pathlib import Path

from pydantic import ValidationError

from ..config import Settings, get_app_dir
from ..conversation import Conversation
from ..images import Image, text_of
from ..models.catalog import model_info
from ..services import Services, open_services
from .prompt import Prompt
from .render import Renderer

HELP = """命令：
  /new              新建会话
  /sessions         列出最近的会话
  /resume ID        恢复会话（ID 可以只写开头几位）
  /rename 标题      重命名当前会话
  /think            切换思考模式
  /history          查看当前会话的消息
  /file 路径        把一个文本文件的内容作为消息发送
  /help             显示本帮助
  /exit             退出（Ctrl+D 也可以）"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Kairos：个人助手")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--continue", dest="resume_latest", action="store_true", help="继续最近的会话")
    group.add_argument("--session", metavar="ID", help="恢复指定会话（ID 可以只写开头几位）")
    args = parser.parse_args()

    try:
        settings = Settings()
    except ValidationError as error:
        raise SystemExit(f"配置不完整，请检查 ~/.kairos/.env：\n{error}") from None
    services = open_services(settings, get_app_dir())
    renderer = Renderer()
    try:
        run(services, renderer, open_conversation(services, args))
    finally:
        renderer.info("正在整理最后一轮的记忆……")
        services.close()
        show_notices(services, renderer)


def open_conversation(services: Services, args: argparse.Namespace) -> Conversation:
    if args.session:
        return Conversation(services, services.sessions.get(args.session))
    if args.resume_latest and (latest := services.sessions.latest()):
        return Conversation(services, latest)
    return Conversation.create(services)


def run(services: Services, renderer: Renderer, conversation: Conversation) -> None:
    prompt = Prompt()
    renderer.info(f"Kairos · {conversation.session.title} · 输入 /help 查看命令")
    while True:
        show_notices(services, renderer)
        try:
            text, images = prompt.read(status=status_line(conversation))
        except KeyboardInterrupt:  # 输入时按 Ctrl+C：清空这次输入，继续
            continue
        except EOFError:  # Ctrl+D
            return

        if text.startswith("/"):
            prompt.images = images  # 命令不消耗待发送的图片
            command, _, argument = text.partition(" ")
            if command == "/exit":
                return
            try:
                conversation = run_command(command, argument.strip(), conversation, services, renderer)
            except (ValueError, OSError) as error:  # 例如会话 ID 不存在、文件读不了
                renderer.error(str(error))
        elif text or images:
            send(conversation, renderer, text, images)


def send(conversation: Conversation, renderer: Renderer, text: str, images: list[Image]) -> None:
    """发送一条消息并显示回答。中断和出错都只结束这一轮，程序继续运行。"""
    try:
        conversation.respond(text, images, on_text=renderer.text_delta, on_message=renderer.message)
        renderer.end_text()
    except KeyboardInterrupt:
        renderer.end_text()
        renderer.info("已中断。")
    except Exception as error:  # 网络、接口等错误：显示出来，让用户决定下一步
        renderer.end_text()
        renderer.error(f"出错了：{type(error).__name__}: {error}")


def run_command(
    command: str, argument: str, conversation: Conversation, services: Services, renderer: Renderer
) -> Conversation:
    """执行一条斜杠命令，返回之后要使用的会话（切换会话的命令会返回新的）。"""
    if command == "/new":
        conversation = Conversation.create(services)
        renderer.info("已新建会话。")
    elif command == "/sessions":
        for session in services.sessions.recent():
            mark = "*" if session.id == conversation.session.id else " "
            renderer.info(f"{mark} {session.id[:8]}  {session.updated_at[:16]}  {session.title}")
    elif command == "/resume":
        conversation = Conversation(services, services.sessions.get(argument))
        renderer.info(f"已恢复：{conversation.session.title}")
    elif command == "/rename":
        conversation.rename(argument)
        renderer.info(f"已重命名为：{argument}")
    elif command == "/think":
        enabled = not conversation.session.thinking
        if conversation.set_thinking(enabled):
            renderer.info(f"思考模式：{'开' if enabled else '关'}")
        else:
            renderer.info("当前模型不支持切换思考模式。")
    elif command == "/history":
        for message in conversation.messages:
            calls = [call["function"]["name"] for call in message.get("tool_calls", [])]
            renderer.info(f"[{message['role']}] {text_of(message['content'] or '')[:200]} {' '.join(calls)}")
    elif command == "/file":
        path = Path(argument).expanduser()
        send(conversation, renderer, f"文件 {path.name} 的内容：\n\n{path.read_text(encoding='utf-8')}", [])
    elif command == "/help":
        renderer.info(HELP)
    else:
        renderer.error(f"未知命令 {command}，输入 /help 查看命令。")
    return conversation


def status_line(conversation: Conversation) -> str:
    """输入框下方的状态：模型、上下文占用、本会话累计用量和费用。"""
    totals = conversation.totals()
    usage = totals.usage
    window = model_info(conversation.session.model).context_window
    context = "上下文 —"
    if totals.last_input_tokens and window:
        context = f"上下文 {totals.last_input_tokens:,}/{window:,}（{totals.last_input_tokens / window:.1%}）"
    thinking = " · 思考开" if conversation.session.thinking else ""
    return (
        f"{conversation.session.model}{thinking} · {context} · "
        f"本会话 输入 {usage.input_tokens:,}（缓存 {usage.cached_input_tokens:,}） "
        f"输出 {usage.output_tokens:,} · ${totals.cost_usd:.4f}"
    )


def show_notices(services: Services, renderer: Renderer) -> None:
    """显示后台记忆提取的结果。"""
    for notice in services.extractor.drain_notices() if services.extractor else []:
        renderer.info(f"（{notice}）")
