"""阶段 1 的最小单工具 Agent 循环。

这里先使用接近 OpenAI 的统一响应形状，方便用 Fake Model 测试消息顺序。
模型适配层和多工具、错误回传、最大轮数会在后续任务中加入。
"""

import json
from collections.abc import Callable, Mapping
from typing import Protocol


class Model(Protocol):
    """模型需要提供的最小接口。"""

    def complete(self, *, messages: list[dict], tools: list[dict]) -> dict: ...


ToolHandler = Callable[..., object]


def _assistant_message(response: dict, tool_calls: list[dict]) -> dict:
    message = {
        "role": "assistant",
        "content": response.get("content"),
        "tool_calls": tool_calls,
    }
    reasoning_content = response.get("reasoning_content")
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return message


def run_single_tool_turn(
    messages: list[dict],
    model: Model,
    tool_handlers: Mapping[str, ToolHandler],
    tool_schemas: list[dict],
) -> tuple[str, list[dict]]:
    """执行可包含多轮、多工具调用的 Agent 回合。

    模型直接回答时函数返回；模型请求工具时，函数执行这一批工具，把 assistant
    工具调用和 tool 结果写回 messages，再次请求模型，直到模型给出最终回答。
    """
    while True:
        response = model.complete(messages=messages, tools=tool_schemas)
        tool_calls = response.get("tool_calls") or []
        if not tool_calls:
            final_content = response.get("content") or ""
            messages.append({"role": "assistant", "content": final_content})
            return final_content, messages

        messages.append(_assistant_message(response, tool_calls))
        for tool_call in tool_calls:
            function = tool_call["function"]
            tool_name = function["name"]
            arguments = json.loads(function.get("arguments", "{}"))
            if not isinstance(arguments, dict):
                raise ValueError("工具参数必须是 JSON 对象")

            result = tool_handlers[tool_name](**arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
