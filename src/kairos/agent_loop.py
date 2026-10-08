"""阶段 1 的最小单工具 Agent 循环。

这里先使用接近 OpenAI 的统一响应形状，方便用 Fake Model 测试消息顺序。
当前支持多工具、跨轮调用、工具错误回传和最大工具响应轮数。
"""

import json
from collections.abc import Callable, Mapping
from typing import Protocol


class Model(Protocol):
    """模型需要提供的最小接口。"""

    def complete(self, *, messages: list[dict], tools: list[dict]) -> dict: ...


ToolHandler = Callable[..., object]


class AgentLoopLimitError(RuntimeError):
    """模型连续请求工具，超过了允许的工具响应轮数。"""


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


def _error_content(error: Exception) -> str:
    """把工具异常转换成模型可以继续处理的 JSON 字符串。"""
    return json.dumps(
        {
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            }
        },
        ensure_ascii=False,
    )


def _execute_tool(tool_call: dict, tool_handlers: Mapping[str, ToolHandler]) -> str:
    """执行单个工具；参数或函数错误都作为工具结果返回。"""
    try:
        function = tool_call["function"]
        tool_name = function["name"]
        arguments = json.loads(function.get("arguments", "{}"))
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是 JSON 对象")

        result = tool_handlers[tool_name](**arguments)
        return json.dumps(result, ensure_ascii=False)
    except Exception as error:
        return _error_content(error)


def run_single_tool_turn(
    messages: list[dict],
    model: Model,
    tool_handlers: Mapping[str, ToolHandler],
    tool_schemas: list[dict],
    *,
    max_tool_rounds: int = 8,
) -> tuple[str, list[dict]]:
    """执行可包含多轮、多工具调用的 Agent 回合。

    模型直接回答时函数返回；模型请求工具时，函数执行这一批工具，把 assistant
    工具调用和 tool 结果写回 messages，再次请求模型，直到模型给出最终回答。
    ``max_tool_rounds`` 只统计包含工具调用的模型响应轮数，同一轮的多个工具
    调用只计一次；超过上限后抛出 AgentLoopLimitError，避免无限循环。
    """
    if max_tool_rounds < 1:
        raise ValueError("max_tool_rounds 必须大于或等于 1")

    tool_rounds = 0
    while True:
        response = model.complete(messages=messages, tools=tool_schemas)
        tool_calls = response.get("tool_calls") or []
        if not tool_calls:
            final_content = response.get("content") or ""
            messages.append({"role": "assistant", "content": final_content})
            return final_content, messages

        if tool_rounds >= max_tool_rounds:
            raise AgentLoopLimitError(f"工具调用超过最大轮数（{max_tool_rounds}）")
        tool_rounds += 1

        messages.append(_assistant_message(response, tool_calls))
        for tool_call in tool_calls:
            if "id" not in tool_call:
                raise ValueError("工具调用缺少 id，无法回传错误结果")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": _execute_tool(tool_call, tool_handlers),
                }
            )
