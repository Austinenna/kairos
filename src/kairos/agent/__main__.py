"""最小可用的对话：只有模型、Agent 循环和待办工具，什么都不保存。

用来单独观察和实验 Agent 循环。运行：uv run python -m kairos.agent
待办存在内存数据库里，退出就消失，不会碰 ~/.kairos 里的数据。
"""

import json
import sqlite3

from kairos_todo.tools import build_tools

from ..config import Settings
from ..models import build_model
from .loop import run_turn

SYSTEM_PROMPT = "你是 Kairos，用户的个人助手。回答简洁直接，使用中文。"


def show_tool_calls(message: dict) -> None:
    """模型决定调用工具时，把调用打印出来，方便观察循环在做什么。"""
    for call in message.get("tool_calls", []):
        function = call["function"]
        print(f"\n  [调用 {function['name']} {function['arguments']}]", flush=True)
    if message["role"] == "tool":
        print(f"  [结果 {json.loads(message['content'])}]", flush=True)


def main() -> None:
    settings = Settings()
    model = build_model(settings)
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    todo = build_tools(connection)
    messages = [{"role": "system", "content": "\n".join([SYSTEM_PROMPT, *todo["instructions"]])}]
    print(f"模型：{model.model}。Ctrl+C 中断回答，Ctrl+D 退出。")

    while True:
        try:
            prompt = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue
        messages.append({"role": "user", "content": prompt})
        print("Kairos> ", end="", flush=True)
        try:
            result = run_turn(
                messages,
                model,
                todo["schemas"],
                todo["handlers"],
                on_text=lambda text: print(text, end="", flush=True),
                on_message=show_tool_calls,
            )
            print(f"\n  （本轮请求模型 {len(result.usages)} 次）")
        except KeyboardInterrupt:
            print("\n[已中断]")


main()
