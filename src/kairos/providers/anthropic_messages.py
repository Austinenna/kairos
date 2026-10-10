"""Anthropic Messages API 适配器。"""

import json
from collections.abc import Mapping
from typing import Any

from .common import field


def anthropic_tools(tools: list[dict]) -> list[dict]:
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


def anthropic_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """把内部消息转换成 Anthropic Messages API 的消息形状。"""

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


def normalize_anthropic_response(response: object) -> dict:
    """把 Anthropic SDK 的 Message 响应转换成 Agent Loop 使用的字典。"""

    content = field(response, "content", []) or []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in content:
        block_type = field(block, "type")
        if block_type == "text":
            text = field(block, "text")
            if text:
                text_parts.append(text)
        elif block_type == "thinking":
            thinking = field(block, "thinking")
            if thinking:
                reasoning_parts.append(thinking)
        elif block_type == "tool_use":
            call_id = field(block, "id")
            name = field(block, "name")
            tool_input = field(block, "input", {})
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

    usage = field(response, "usage")
    if usage is not None:
        input_tokens = field(usage, "input_tokens")
        output_tokens = field(usage, "output_tokens")
        result["usage"] = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": (input_tokens or 0) + (output_tokens or 0),
        }
    stop_reason = field(response, "stop_reason")
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
        system, claude_messages = anthropic_messages(messages)
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": claude_messages,
            "tools": anthropic_tools(tools),
        }
        if system is not None:
            request["system"] = system
        response = self.client.messages.create(**request)
        return normalize_anthropic_response(response)
