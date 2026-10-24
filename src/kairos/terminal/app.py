"""终端主循环：读输入 → 执行命令或发给 Conversation → 显示结果。

运行：uv run kairos [--continue | --session ID] [--channels terminal,weixin]
"""

import argparse
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import ValidationError

from ..config import Settings, get_app_dir
from ..conversation import Conversation
from ..images.prepare import Image, text_of
from ..models.catalog import model_info
from ..services import Services, open_services
from ..wechat.runner import WeixinRunner
from .interrupt import EscInterrupt
from .prompt import Prompt
from .render import Renderer

CHANNEL_NAMES = {"terminal": "终端", "weixin": "微信"}
EXIT_WINDOW = 2  # 两次 Ctrl+C 间隔不超过这么多秒就退出

HELP = """命令：
  /new              新建会话
  /sessions         列出最近的会话
  /resume ID        恢复会话（ID 可以只写开头几位）
  /rename 标题      重命名当前会话
  /think            切换思考模式
  /history          查看当前会话的消息
  /file 路径        把一个文本文件的内容作为消息发送
  /help             显示本帮助
  /exit             退出（Ctrl+D、连按两次 Ctrl+C 也可以）
生成回答时按 Esc 中断。"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Kairos：个人助手")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--continue", dest="resume_latest", action="store_true", help="继续最近的会话")
    group.add_argument("--session", metavar="ID", help="恢复指定会话（ID 可以只写开头几位）")
    parser.add_argument("--channels", default="terminal,weixin", help="启动哪些入口，默认 terminal,weixin")
    args = parser.parse_args()
    channels = set(args.channels.split(","))

    try:
        settings = Settings()
    except ValidationError as error:
        raise SystemExit(f"配置不完整，请检查 ~/.kairos/.env：\n{error}") from None
    services = open_services(settings, get_app_dir())
    renderer = Renderer()
    weixin = start_weixin(services, renderer) if "weixin" in channels else None
    try:
        if "terminal" in channels:
            run(services, renderer, open_conversation(services, args), weixin)
        elif weixin:
            wait_for_weixin(services, renderer, weixin)
    finally:
        if weixin:
            weixin.stop()
        renderer.info("正在整理最后一轮的记忆……")
        with services.lock:  # 等微信正在处理的那条消息结束
            services.close()
        show_notices(services, renderer, weixin)


def start_weixin(services: Services, renderer: Renderer) -> WeixinRunner | None:
    """启动微信入口。没登录或凭据读不了时只提示，终端照常使用。"""
    data_dir = get_app_dir() / "channels" / "weixin"
    if not (data_dir / "account.json").exists():
        renderer.info("微信未登录，本次只启动终端。登录：uv run --with 'qrcode[pil]' python -m kairos.wechat")
        return None
    try:
        weixin = WeixinRunner(services, data_dir)
    except (OSError, ValueError, KeyError) as error:
        renderer.error(f"微信入口没有启动：{type(error).__name__}: {error}")
        return None
    weixin.start()
    renderer.info("微信入口已启动。")
    return weixin


def wait_for_weixin(services: Services, renderer: Renderer, weixin: WeixinRunner) -> None:
    """只启动微信时，主线程在这里显示状态，按 Ctrl+C 退出。"""
    renderer.info("Kairos 正在运行微信入口，按 Ctrl+C 退出。")
    try:
        while weixin.thread.is_alive():
            show_notices(services, renderer, weixin)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass


def open_conversation(services: Services, args: argparse.Namespace) -> Conversation:
    if args.session:
        return Conversation(services, services.sessions.get(args.session))
    if args.resume_latest and (latest := services.sessions.latest()):
        return Conversation(services, latest)
    return Conversation.create(services)


def run(services: Services, renderer: Renderer, conversation: Conversation, weixin: WeixinRunner | None) -> None:
    prompt = Prompt()
    ctrl_c = DoubleCtrlC()
    renderer.info(f"Kairos · {conversation.session.title} · 输入 /help 查看命令")
    while True:
        show_notices(services, renderer, weixin)
        status = "微信那边正在回复……"
        if services.lock.acquire(blocking=False):  # 微信正在用数据库时不等它，状态行先显示提示
            try:
                status = status_line(conversation)
            finally:
                services.lock.release()
        try:
            text, images = prompt.read(status=status)
        except KeyboardInterrupt:  # 输入时按 Ctrl+C：清空这次输入；连按两次退出
            if ctrl_c.pressed():
                return
            prompt.notice = "2 秒内再按一次 Ctrl+C 退出"
            continue
        except EOFError:  # Ctrl+D
            return

        if text.startswith("/"):
            prompt.images = images  # 命令不消耗待发送的图片
            command, _, argument = text.partition(" ")
            if command == "/exit":
                return
            with taking_turn(services, renderer):
                try:
                    conversation = run_command(command, argument.strip(), conversation, services, renderer)
                except (ValueError, OSError) as error:  # 例如会话 ID 不存在、文件读不了
                    renderer.error(str(error))
        elif text or images:
            with taking_turn(services, renderer), EscInterrupt() as esc:
                interrupted = send(conversation, renderer, text, images)
            if interrupted and not esc.pressed:  # 生成中按的是 Ctrl+C：也算一次，紧接着再按一次就退出
                if ctrl_c.pressed():
                    return
                prompt.notice = "2 秒内再按一次 Ctrl+C 退出"


class DoubleCtrlC:
    """记录 Ctrl+C：两次间隔不超过 EXIT_WINDOW 秒，就表示要退出。"""

    def __init__(self):
        self.last = 0.0

    def pressed(self) -> bool:
        """按了一次 Ctrl+C。返回 True 表示和上一次连在一起，应该退出。"""
        now = time.monotonic()
        twice = now - self.last < EXIT_WINDOW
        self.last = now
        return twice


@contextmanager
def taking_turn(services: Services, renderer: Renderer) -> Iterator[None]:
    """拿到 services.lock 再处理终端输入；微信正在回复时先提示，等它结束。"""
    if not services.lock.acquire(blocking=False):
        renderer.info("微信那边正在回复，稍等……")
        services.lock.acquire()
    try:
        yield
    finally:
        services.lock.release()


def send(conversation: Conversation, renderer: Renderer, text: str, images: list[Image]) -> bool:
    """发送一条消息并显示回答。中断和出错都只结束这一轮，程序继续运行。返回是否被中断。"""
    try:
        conversation.respond(text, images, on_text=renderer.text_delta, on_message=renderer.message)
        renderer.end_text()
    except KeyboardInterrupt:
        renderer.end_text()
        renderer.info("已中断。")
        return True
    except Exception as error:  # 网络、接口等错误：显示出来，让用户决定下一步
        renderer.end_text()
        renderer.error(f"出错了：{type(error).__name__}: {error}")
    return False


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
            channel = CHANNEL_NAMES.get(session.channel, session.channel)
            renderer.info(f"{mark} {session.id[:8]}  {session.updated_at[:16]}  {channel}  {session.title}")
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
        used = totals.last_input_tokens
        context = f"上下文 {short(used)}/{short(window)} ({used / window:.1%})"
    thinking = " · 思考开" if conversation.session.thinking else ""
    return (
        f"{conversation.session.model}{thinking} · {context} · "
        f"输入 {short(usage.input_tokens)} 缓存 {short(usage.cached_input_tokens)} "
        f"输出 {short(usage.output_tokens)} · ${totals.cost_usd:.4f}"
    )


def short(n: int) -> str:
    """把 token 数缩写，状态行才放得下一行：3666 → 3.7k，1048576 → 1M。"""
    for unit, size in (("M", 1_000_000), ("k", 1_000)):
        if n >= size:
            return f"{n / size:.1f}".removesuffix(".0") + unit
    return str(n)


def show_notices(services: Services, renderer: Renderer, weixin: WeixinRunner | None = None) -> None:
    """显示后台记忆提取和微信入口的状态。"""
    notices = services.extractor.drain_notices() if services.extractor else []
    notices += weixin.drain_notices() if weixin else []
    for notice in notices:
        renderer.info(f"（{notice}）")
