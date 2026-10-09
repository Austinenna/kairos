"""把不同模型的请求和响应转换成 Agent 循环使用的统一格式。

``agent_loop`` 只依赖一个很小的 ``Model.complete`` 接口。本模块把 OpenAI 兼容格式
和 Anthropic Messages 格式都转换成同一套内部字典，
因此循环本身不需要知道供应商差异。
"""

import json
from collections.abc import Mapping
from typing import Any

from openai import OpenAI


def _field(value: object, name: str, default: Any = None) -> Any:
    """同时读取 SDK 对象和测试中使用的字典。"""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalize_tool_call(tool_call: object) -> dict:
    function = _field(tool_call, "function", {})
    name = _field(function, "name")
    call_id = _field(tool_call, "id")
    if not name or not call_id:
        raise ValueError("模型返回的工具调用缺少 id 或 name")

    arguments = _field(function, "arguments", "{}")
    if not isinstance(arguments, str):
        raise TypeError("模型返回的工具参数必须是 JSON 字符串")

    return {
        "id": call_id,
        "type": _field(tool_call, "type", "function"),
        "function": {"name": name, "arguments": arguments},
    }


class OpenAICompatibleModel:
    """调用 OpenAI 兼容接口，并返回循环使用的统一字典。"""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        extra_body: dict[str, Any] | None = None,
    ):
        self.model = model
        self.extra_body = extra_body
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, *, messages: list[dict], tools: list[dict]) -> dict:
        request = {"model": self.model, "messages": messages, "tools": tools}
        if self.extra_body is not None:
            request["extra_body"] = self.extra_body
        response = self.client.chat.completions.create(**request)
        choices = _field(response, "choices", [])
        if not choices:
            raise ValueError("模型响应没有 choices")

        message = _field(choices[0], "message")
        if message is None:
            raise ValueError("模型响应没有 message")

        tool_calls = [_normalize_tool_call(call) for call in (_field(message, "tool_calls", []) or [])]
        result = {
            "content": _field(message, "content"),
            "tool_calls": tool_calls,
        }
        usage = _field(response, "usage")
        if usage is not None:
            result["usage"] = {
                "prompt_tokens": _field(usage, "prompt_tokens"),
                "completion_tokens": _field(usage, "completion_tokens"),
                "total_tokens": _field(usage, "total_tokens"),
            }
        finish_reason = _field(choices[0], "finish_reason")
        if finish_reason is not None:
            result["finish_reason"] = finish_reason
        reasoning_content = _field(message, "reasoning_content")
        if reasoning_content is not None:
            result["reasoning_content"] = reasoning_content
        return result


def _anthropic_tools(tools: list[dict]) -> list[dict]:
    """把 OpenAI 风格的函数工具定义转换成 Anthropic 的工具定义。"""
    converted = []
    for tool in tools:
        function = tool.get("function", {})
        converted.append(
            {
                "name": function["name"],
                "description": function.get("description", ""),
                "input_schema": function.get(
                    "parameters", {"type": "object", "properties": {}, "additionalProperties": False}
                ),
            }
        )
    return converted


def _anthropic_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """把循环内部的消息转换成 Anthropic Messages API 的消息形状。

    Anthropic 把 system 单独作为请求参数，并要求工具结果放在 user 消息的
    ``tool_result`` 内容块中；内部循环仍保留 OpenAI 风格的 ``tool`` 消息，
    这样两种模型可以共用同一套循环和测试。
    """
    system_parts: list[str] = []
    converted: list[dict] = []
    pending_tool_results: list[dict] = []

    def flush_tool_results() -> None:
        if pending_tool_results:
            converted.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for message in messages:
        role = message.get("role")
        if role == "system":
            content = message.get("content")
            if content:
                system_parts.append(str(content))
            continue

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not tool_call_id:
                raise ValueError("工具结果缺少 tool_call_id")
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": str(message.get("content", "")),
                }
            )
            continue

        flush_tool_results()
        if role == "user":
            converted.append({"role": "user", "content": message.get("content") or ""})
            continue

        if role == "assistant":
            content_blocks: list[dict] = []
            content = message.get("content")
            if content:
                content_blocks.append({"type": "text", "text": str(content)})
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function", {})
                arguments = json.loads(function.get("arguments", "{}"))
                if not isinstance(arguments, dict):
                    raise ValueError("Anthropic 工具调用参数必须是 JSON 对象")
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": function["name"],
                        "input": arguments,
                    }
                )
            converted.append({"role": "assistant", "content": content_blocks or ""})
            continue

        raise ValueError(f"不支持的消息角色：{role}")

    flush_tool_results()
    return ("\n\n".join(system_parts) or None), converted


def _normalize_anthropic_response(response: object) -> dict:
    """把 Anthropic SDK 的 Message 响应转换成循环使用的统一字典。"""
    content = _field(response, "content", []) or []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in content:
        block_type = _field(block, "type")
        if block_type == "text":
            text = _field(block, "text")
            if text:
                text_parts.append(text)
        elif block_type == "thinking":
            thinking = _field(block, "thinking")
            if thinking:
                reasoning_parts.append(thinking)
        elif block_type == "tool_use":
            call_id = _field(block, "id")
            name = _field(block, "name")
            tool_input = _field(block, "input", {})
            if not call_id or not name or not isinstance(tool_input, Mapping):
                raise ValueError("Anthropic 工具调用缺少 id、name 或对象参数")
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(dict(tool_input), ensure_ascii=False),
                    },
                }
            )
        elif block_type not in {"redacted_thinking"}:
            raise ValueError(f"Anthropic 返回了不支持的内容块：{block_type}")

    result = {"content": "\n".join(text_parts) or None, "tool_calls": tool_calls}
    if reasoning_parts:
        result["reasoning_content"] = "\n".join(reasoning_parts)

    usage = _field(response, "usage")
    if usage is not None:
        input_tokens = _field(usage, "input_tokens")
        output_tokens = _field(usage, "output_tokens")
        result["usage"] = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": (input_tokens or 0) + (output_tokens or 0),
        }
    stop_reason = _field(response, "stop_reason")
    if stop_reason is not None:
        result["finish_reason"] = stop_reason
    return result


class AnthropicMessagesModel:
    """使用 Anthropic Messages API，并输出统一的内部字典。"""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.max_tokens = max_tokens
        if client is not None:
            self.client = client
            return
        try:
            from anthropic import Anthropic
        except ModuleNotFoundError as error:
            raise RuntimeError("使用 AnthropicMessagesModel 需要先安装 anthropic 包") from error
        self.client = Anthropic(api_key=api_key, base_url=base_url)

    def complete(self, *, messages: list[dict], tools: list[dict]) -> dict:
        system, claude_messages = _anthropic_messages(messages)
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": claude_messages,
            "tools": _anthropic_tools(tools),
        }
        if system is not None:
            request["system"] = system
        response = self.client.messages.create(**request)
        return _normalize_anthropic_response(response)
