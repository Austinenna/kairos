"""离线观察工具报错行为的实验脚本。

这个脚本故意伪造模型返回的工具调用，不访问真实 API、不连接 SQLite。
它用于在修改错误回传逻辑前，记录当前程序遇到错误时的基线行为。

运行：

    uv run python scripts/experiment_tool_errors.py
    uv run python scripts/experiment_tool_errors.py --case invalid-json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

from kairos.agent_loop import run_single_tool_turn


class FakeModel:
    """每次返回同一个工具调用，并记录模型被调用了几次。"""

    def __init__(self, response: dict[str, Any]):
        self.response = response
        self.calls: list[list[dict]] = []

    def complete(self, *, messages: list[dict], tools: list[dict]) -> dict:
        self.calls.append(list(messages))
        return self.response


def _tool_call(name: str, arguments: str) -> dict[str, Any]:
    return {
        "content": None,
        "tool_calls": [
            {
                "id": "call_demo",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _broken_handler() -> dict[str, bool]:
    raise ValueError("这是实验用的工具错误")


def _cases() -> dict[str, tuple[str, dict[str, Any], dict[str, Callable[..., object]]]]:
    return {
        "invalid-json": (
            "非法 JSON 参数",
            _tool_call("demo", "{not-json"),
            {"demo": lambda: {"ok": True}},
        ),
        "unknown-tool": (
            "未知工具名称",
            _tool_call("missing_tool", "{}"),
            {"demo": lambda: {"ok": True}},
        ),
        "handler-error": (
            "工具函数主动报错",
            _tool_call("demo", "{}"),
            {"demo": _broken_handler},
        ),
    }


def run_case(case_name: str) -> None:
    title, response, handlers = _cases()[case_name]
    model = FakeModel(response)
    messages = [{"role": "user", "content": title}]

    print(f"=== {case_name}：{title} ===")
    try:
        answer, messages = run_single_tool_turn(messages, model, handlers, [])
    except Exception as error:
        print(f"异常：{type(error).__name__}: {error}")
    else:
        print(f"返回：{answer}")
    print(f"model.complete 调用次数：{len(model.calls)}")
    print("停止时的 messages：")
    print(json.dumps(messages, ensure_ascii=False, indent=2))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="离线观察 Kairos 当前工具报错行为")
    parser.add_argument(
        "--case",
        choices=["all", *_cases()],
        default="all",
        help="只运行一个错误案例，默认运行全部案例",
    )
    args = parser.parse_args()

    case_names = _cases() if args.case == "all" else {args.case: _cases()[args.case]}
    for case_name in case_names:
        run_case(case_name)


if __name__ == "__main__":
    main()
