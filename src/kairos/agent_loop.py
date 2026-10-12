"""Kairos 的 Agent 循环。

这里先使用接近 OpenAI 的统一响应形状，方便用 Fake Model 测试消息顺序。
当前支持多工具、跨轮调用、工具错误回传和最大工具响应轮数。
"""

import json
from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any, Protocol, runtime_checkable


class Model(Protocol):
    """模型需要提供的最小接口。"""

    def complete(self, *, messages: list[dict], tools: list[dict]) -> dict: ...


@runtime_checkable
class StreamingModel(Protocol):
    """同步消费一次模型流；正文即时回调，完整响应在流正常结束后返回。"""

    def stream(self, *, messages: list[dict], tools: list[dict], on_text: Callable[[str], None]) -> dict: ...


ToolHandler = Callable[..., object]
EventLogger = Callable[[dict[str, Any]], None]


class AgentLoopLimitError(RuntimeError):
    """模型连续请求工具，超过了允许的工具响应轮数。"""


def _assistant_message(response: dict, tool_calls: list[dict]) -> dict:
    message = {
        "role": "assistant",
        "content": response.get("content"),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    reasoning_content = response.get("reasoning_content")
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    # Anthropic 的内容块包含顺序和思考签名，必须原样用于后续请求。
    if "provider_content" in response:
        message["provider_content"] = response["provider_content"]
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


def run_agent_turn(
    messages: list[dict],
    model: Model,
    tool_handlers: Mapping[str, ToolHandler],
    tool_schemas: list[dict],
    *,
    max_tool_rounds: int = 8,
    logger: EventLogger | None = None,
    on_event: EventLogger | None = None,
) -> tuple[str, list[dict]]:
    """执行可包含多轮、多工具调用的 Agent 回合。

    模型直接回答时函数返回；模型请求工具时，函数执行这一批工具，把 assistant
    工具调用和 tool 结果写回 messages，再次请求模型，直到模型给出最终回答。
    ``max_tool_rounds`` 只统计包含工具调用的模型响应轮数，同一轮的多个工具
    调用只计一次；超过上限后抛出 AgentLoopLimitError，避免无限循环。
    传入 on_event 时使用模型流式接口；不支持流式的模型仍可返回完整响应。
    text_delta 仅交给展示回调，日志仍以完整模型响应和工具结果为单位。
    """
    if max_tool_rounds < 1:
        raise ValueError("max_tool_rounds 必须大于或等于 1")

    tool_rounds = 0
    request_number = 0

    def emit(event: dict[str, Any]) -> None:
        if logger:
            logger(event)
        if on_event:
            on_event(event)

    def text_delta(text: str) -> None:
        if text and on_event:
            on_event({"event": "text_delta", "request_number": request_number, "text": text})

    while True:
        request_number += 1
        emit(
            {
                "event": "model_request",
                "request_number": request_number,
                "tool_round": tool_rounds,
                "messages": list(messages),
                "tools": list(tool_schemas),
            }
        )

        model_started = perf_counter()
        try:
            if on_event is not None and isinstance(model, StreamingModel):
                response = model.stream(messages=messages, tools=tool_schemas, on_text=text_delta)
            else:
                response = model.complete(messages=messages, tools=tool_schemas)
                text_delta(response.get("content") or "")
        except (Exception, KeyboardInterrupt) as error:
            emit(
                {
                    "event": "model_error",
                    "request_number": request_number,
                    "tool_round": tool_rounds,
                    "error": {"type": type(error).__name__, "message": str(error)},
                    "duration_ms": round((perf_counter() - model_started) * 1000, 3),
                }
            )

            raise
        emit(
            {
                "event": "model_response",
                "request_number": request_number,
                "tool_round": tool_rounds,
                "response": response,
                "usage": response.get("usage"),
                "duration_ms": round((perf_counter() - model_started) * 1000, 3),
            }
        )
        tool_calls = response.get("tool_calls") or []
        if not tool_calls:
            final_content = response.get("content") or ""
            messages.append(_assistant_message({**response, "content": final_content}, []))
            return final_content, messages

        if tool_rounds >= max_tool_rounds:
            raise AgentLoopLimitError(f"工具调用超过最大轮数（{max_tool_rounds}）")
        tool_rounds += 1

        # 先检查整批调用的关联编号，避免执行到一半才发现无法配对的调用。
        call_ids = [call.get("id") for call in tool_calls]
        if any(not call_id for call_id in call_ids) or len(set(call_ids)) != len(call_ids):
            raise ValueError("工具调用缺少或重复 id，无法回传结果")
        batch_start = len(messages)
        active_call: str | None = None
        try:
            messages.append(_assistant_message(response, tool_calls))
            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                result_started = perf_counter()
                active_call = tool_call["id"]
                result_content = _execute_tool(tool_call, tool_handlers)
                messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result_content})
                active_call = None
                emit(
                    {
                        "event": "tool_result",
                        "request_number": request_number,
                        "tool_round": tool_rounds,
                        "tool_call_id": tool_call["id"],
                        "tool_name": function.get("name"),
                        "arguments": function.get("arguments"),
                        "result": result_content,
                        "duration_ms": round((perf_counter() - result_started) * 1000, 3),
                    }
                )
        except (Exception, KeyboardInterrupt):
            # 执行、记录或展示任一步中断时，保留已记录的结果并补齐未完成调用。
            # 不能删除历史来“撤销”已经发生的副作用，也不能留下未配对的工具消息。
            completed = {m["tool_call_id"] for m in messages[batch_start + 1 :] if m["role"] == "tool"}
            for pending in tool_calls if len(messages) > batch_start else []:
                if pending["id"] in completed:
                    continue
                message = (
                    "工具执行被中断，结果未确认；可能已产生副作用，请先查询状态，不要直接重试。"
                    if pending["id"] == active_call
                    else "本次工具调用因回合中断而未执行。"
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": pending["id"],
                        "content": _error_content(InterruptedError(message)),
                    }
                )
            raise
