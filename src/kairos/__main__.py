"""新版 Kairos 的临时纯文字入口，用来试用重构中的代码：uv run python -m kairos

会话保存在 ~/.kairos/kairos.db（与旧版的 kairos.sqlite3 分开）。
第 4 步会换成正式的终端界面；旧版 `uv run kairos` 在那之前照常可用。
"""

import json

from kairos_todo.tools import build_tools

from .config import Settings, get_app_dir
from .conversation import Conversation
from .storage import SessionStore, open_db

HELP = "命令：/new 新会话  /sessions 列表  /resume ID 恢复  /rename 标题  /think 切换思考  /usage 用量  /exit 退出"


def show_tool_calls(message: dict) -> None:
    for call in message.get("tool_calls", []):
        print(f"\n  [调用 {call['function']['name']} {call['function']['arguments']}]", flush=True)


def main() -> None:
    settings = Settings()
    db = open_db(get_app_dir() / "kairos.db")
    store = SessionStore(db)
    todo = build_tools(db)
    tools = (todo["schemas"], todo["handlers"])
    conversation = Conversation.create(store, settings, *tools)
    print(f"Kairos（{conversation.session.model}）。{HELP}")

    while True:
        try:
            line = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        command, _, argument = line.partition(" ")
        if not line:
            continue
        elif command == "/exit":
            break
        elif command == "/new":
            conversation = Conversation.create(store, settings, *tools)
            print("已新建会话。")
        elif command == "/sessions":
            for session in store.recent():
                mark = "*" if session.id == conversation.session.id else " "
                print(f"{mark} {session.id[:8]}  {session.updated_at[:16]}  {session.title}")
        elif command == "/resume":
            try:
                conversation = Conversation(store, store.get(argument), settings, *tools)
            except ValueError as error:
                print(error)
                continue
            print(f"已恢复：{conversation.session.title}（{len(conversation.messages)} 条消息）")
        elif command == "/rename":
            conversation.rename(argument)
        elif command == "/think":
            enabled = not conversation.session.thinking
            if conversation.set_thinking(enabled):
                print(f"思考模式：{'开' if enabled else '关'}")
            else:
                print("当前模型不支持切换思考模式。")
        elif command == "/usage":
            totals = conversation.totals()
            print(json.dumps({**vars(totals.usage), "cost_usd": round(totals.cost_usd, 6)}, ensure_ascii=False))
        else:
            print("Kairos> ", end="", flush=True)
            try:
                conversation.respond(
                    line, on_text=lambda text: print(text, end="", flush=True), on_message=show_tool_calls
                )
            except KeyboardInterrupt:
                print("\n[已中断]")
            print()


main()
