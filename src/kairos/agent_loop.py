"""阶段 1 的手写 Agent 循环。

循环的基本协议是：请求模型 → 执行这一轮的全部工具 → 把结果写回消息 →
继续请求，直到模型直接回答或达到工具调用轮数上限。模型请求本身的异常继续
向上抛出；单个工具的参数和执行异常则作为该工具的结果交回模型，避免一个
工具失败就丢掉整轮对话。
"""

import json
from collections.abc import Callable, Mapping
from typing import Protocol


class Model(Protocol):
    """模型需要提供的最小接口。"""

    def complete(self, *, messages: list[dict], tools: list[dict]) -> dict: ...


ToolHandler = Callable[..., object]


class AgentLoopLimitError(RuntimeError):
    """模型持续请求工具，超过了允许的工具调用轮数。"""


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
    """把工具错误转换为短小、稳定、可供模型理解的 JSON。"""
    return json.dumps(
        {
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            }
        },
        ensure_ascii=False,
    )


def _tool_result(tool_call: object, tool_handlers: Mapping[str, ToolHandler]) -> str:
    """执行一个工具调用，成功或失败都返回可序列化的字符串结果。

    工具调用的 id 和函数名是协议字段，缺少它们时无法生成合法的 tool 消息，
    因此这类结构性响应直接报错；JSON 参数、未知工具和 handler 异常都回传给模型。
    """
    if not isinstance(tool_call, Mapping):
        raise ValueError("工具调用必须是对象")

    call_id = tool_call.get("id")
    function = tool_call.get("function")
    if not call_id or not isinstance(function, Mapping):
        raise ValueError("模型返回的工具调用缺少 id 或 function")

    tool_name = function.get("name")
    if not tool_name:
        raise ValueError("模型返回的工具调用缺少函数名")

    try:
        arguments = json.loads(function.get("arguments", "{}"))
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是 JSON 对象")
        handler = tool_handlers.get(tool_name)
        if handler is None:
            raise ValueError(f"未知工具：{tool_name}")
        result = handler(**arguments)
        return json.dumps(result, ensure_ascii=False)
    except Exception as error:
        return _error_content(error)


def run_tool_loop(
    messages: list[dict],
    model: Model,
    tool_handlers: Mapping[str, ToolHandler],
    tool_schemas: list[dict],
    *,
    max_tool_rounds: int = 8,
) -> tuple[str, list[dict]]:
    """运行待办 Agent 循环，支持多工具调用、错误回传和轮数上限。

    max_tool_rounds 统计包含工具调用的模型响应轮数；同一响应中的多个工具
    调用只计一轮。达到上限后保留已有消息并抛出 AgentLoopLimitError，
    防止模型在工具调用之间无限循环。
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
            if response.get("finish_reason") == "length":
                final_content += "\n\n（模型输出达到长度上限，以上内容可能不完整。）"
                messages[-1]["content"] = final_content
            return final_content, messages

        if tool_rounds >= max_tool_rounds:
            raise AgentLoopLimitError(f"工具调用超过最大轮数（{max_tool_rounds}）")
        tool_rounds += 1

        messages.append(_assistant_message(response, tool_calls))
        for tool_call in tool_calls:
            # 缺少 id/function 时无法构造符合协议的 tool 消息，保留 fail-fast 行为。
            tool_id = tool_call.get("id") if isinstance(tool_call, Mapping) else None
            if not tool_id:
                raise ValueError("模型返回的工具调用缺少 id")
            result = _tool_result(tool_call, tool_handlers)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result,
                }
            )


def run_single_tool_turn(
    messages: list[dict],
    model: Model,
    tool_handlers: Mapping[str, ToolHandler],
    tool_schemas: list[dict],
) -> tuple[str, list[dict]]:
    """兼容旧名称；现在执行完整的、有上限的工具循环。"""
    return run_tool_loop(messages, model, tool_handlers, tool_schemas)
