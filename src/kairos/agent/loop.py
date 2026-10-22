"""Agent 循环：请求模型 → 执行工具 → 把结果交回模型，直到模型给出不带工具调用的回答。"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from ..models import Model, Usage

ToolHandler = Callable[..., object]


class TooManyRequests(RuntimeError):
    """一轮对话里请求模型的次数超过上限，通常是模型在反复调用工具。"""


@dataclass
class TurnResult:
    content: str  # 模型最后的回答
    usages: list[Usage] = field(default_factory=list)  # 本轮每次模型请求的用量


def run_turn(
    messages: list[dict],
    model: Model,
    tools: list[dict],
    handlers: dict[str, ToolHandler],
    *,
    prepare: Callable[[list[dict]], list[dict]] | None = None,
    on_text: Callable[[str], None] | None = None,
    on_message: Callable[[dict], None] | None = None,
    max_requests: int = 10,
) -> TurnResult:
    """执行一轮对话。调用前，用户消息应该已经在 messages 末尾。

    参数:
        messages: 会话历史，本函数会原地追加助手消息和工具结果。
        model: 任何实现了 stream() 的模型适配器。
        tools: 发给模型的工具定义（OpenAI 格式）。
        handlers: 工具名 → 执行函数。
        prepare: 每次请求前把历史加工成请求视图（加记忆、时间等），不改原历史。
        on_text: 正文片段回调，用于流式显示。
        on_message: 每追加一条消息就调用一次，用于即时保存。
        max_requests: 本轮最多请求模型几次。

    行为说明:
        1. 工具按模型给出的顺序串行执行；出错时把错误作为工具结果交回模型，让它自己处理。
        2. 每条消息追加后立刻交给 on_message 保存，这样已经执行的工具不会因为后面出错而丢失记录。
        3. 执行工具时被 Ctrl+C 中断，会给这批里没有结果的调用补一条说明。
           协议要求每个工具调用都有对应结果，否则下一次请求会被拒绝。
    """

    def append(message: dict) -> None:
        messages.append(message)
        if on_message:
            on_message(message)

    usages: list[Usage] = []
    for _ in range(max_requests):
        request = prepare(messages) if prepare else messages
        response = model.stream(request, tools, on_text)
        if response.usage:
            usages.append(response.usage)
        append(response.to_message())

        if not response.tool_calls:
            return TurnResult(content=response.content, usages=usages)

        for index, call in enumerate(response.tool_calls):
            try:
                result = execute_tool(call, handlers)
            except KeyboardInterrupt:
                for unfinished in response.tool_calls[index:]:
                    append(_tool_message(unfinished, _error("用户中断：工具未执行，或已执行但结果未知。")))
                raise
            append(_tool_message(call, result))

    raise TooManyRequests(f"本轮已请求模型 {max_requests} 次仍未结束")


def execute_tool(call: dict, handlers: dict[str, ToolHandler]) -> str:
    """执行一个工具调用，返回交给模型的 JSON 字符串；任何错误都变成错误结果。"""
    name = call["function"]["name"]
    if name not in handlers:
        return _error(f"没有名为 {name} 的工具，可用工具：{', '.join(handlers)}")
    try:
        arguments = json.loads(call["function"]["arguments"] or "{}")
        result = handlers[name](**arguments)
    except Exception as error:
        return _error(f"{type(error).__name__}: {error}")
    # default=str：结果里有日期等对象也能序列化，避免工具已执行却报成失败。
    return json.dumps(result, ensure_ascii=False, default=str)


def _tool_message(call: dict, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call["id"], "content": content}


def _error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)
