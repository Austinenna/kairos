"""OpenAI Chat Completions 兼容模型适配器。"""

import json
from collections.abc import Callable
from typing import Any

from openai import OpenAI

from .common import field, normalize_tool_call


def openai_user_content(content: str | list[dict]) -> str | list[dict]:
    """只转换用户内容块，内部附件元数据不进入模型协议。"""

    if not isinstance(content, list):
        return content
    converted = []
    for block in content:
        if block.get("type") == "image":
            converted.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{block['mime_type']};base64,{block['data']}"},
                }
            )
        elif block.get("type") == "text":
            converted.append({"type": "text", "text": block["text"]})
        else:
            raise ValueError("OpenAI 用户消息包含不支持的内容块")
    return converted


def openai_messages(messages: list[dict]) -> list[dict]:
    """保持 assistant 与 tool 消息字段，只投影用户的图文内容。"""

    return [
        {**message, "content": openai_user_content(message.get("content", ""))}
        if message.get("role") == "user"
        else dict(message)
        for message in messages
    ]


class OpenAICompatibleModel:
    """调用 OpenAI 兼容接口，并返回 Agent Loop 使用的统一字典。"""

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
        request = {"model": self.model, "messages": openai_messages(messages), "tools": tools}
        if self.extra_body is not None:
            request["extra_body"] = self.extra_body
        response = self.client.chat.completions.create(**request)
        choices = field(response, "choices", [])
        if not choices:
            raise ValueError("模型响应没有 choices")

        message = field(choices[0], "message")
        if message is None:
            raise ValueError("模型响应没有 message")

        tool_calls = [normalize_tool_call(call) for call in (field(message, "tool_calls", []) or [])]
        result = {
            "content": field(message, "content"),
            "tool_calls": tool_calls,
        }
        usage = field(response, "usage")
        if usage is not None:
            result["usage"] = {
                "prompt_tokens": field(usage, "prompt_tokens"),
                "completion_tokens": field(usage, "completion_tokens"),
                "total_tokens": field(usage, "total_tokens"),
            }
        finish_reason = field(choices[0], "finish_reason")
        if finish_reason is not None:
            result["finish_reason"] = finish_reason
        reasoning_content = field(message, "reasoning_content")
        if reasoning_content is not None:
            result["reasoning_content"] = reasoning_content
        return result

    def stream(self, *, messages: list[dict], tools: list[dict], on_text: Callable[[str], None]) -> dict:
        """逐片交付正文，完整收完响应后才返回可供 Agent 执行的工具调用。"""

        request: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages(messages),
            "tools": tools,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.extra_body is not None:
            request["extra_body"] = self.extra_body

        response_stream = self.client.chat.completions.create(**request)
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_parts: dict[int, dict[str, list[str]]] = {}
        finish_reason: str | None = None
        usage: dict[str, Any] | None = None
        try:
            for chunk in response_stream:
                chunk_usage = field(chunk, "usage")
                if chunk_usage is not None:
                    usage = _stream_usage(chunk_usage)
                for choice in field(chunk, "choices", []) or []:
                    if field(choice, "index", 0) != 0:
                        continue
                    delta = field(choice, "delta")
                    text = field(delta, "content")
                    reasoning = field(delta, "reasoning_content")
                    calls = field(delta, "tool_calls", []) or []
                    if finish_reason is not None and (text or reasoning or calls):
                        raise ValueError("模型在 finish_reason 之后仍返回内容")
                    if text is not None:
                        if not isinstance(text, str):
                            raise ValueError("模型流式正文必须是字符串")
                        text_parts.append(text)
                        if text:
                            on_text(text)
                    if reasoning is not None:
                        if not isinstance(reasoning, str):
                            raise ValueError("模型流式思考内容必须是字符串")
                        reasoning_parts.append(reasoning)
                    if field(delta, "function_call") is not None:
                        raise ValueError("模型返回了不支持的旧式 function_call")
                    for call in calls:
                        _append_tool_delta(tool_parts, call)
                    reason = field(choice, "finish_reason")
                    if reason is not None:
                        if finish_reason is not None and finish_reason != reason:
                            raise ValueError("模型返回了相互冲突的 finish_reason")
                        finish_reason = reason

            # finish_reason 后仍可能有 choices=[] 的用量帧，必须读到流结束再校验和返回。
            if finish_reason is None:
                raise ValueError("模型流未完整结束：缺少 finish_reason")
            if finish_reason not in {"stop", "tool_calls", "length", "content_filter"}:
                raise ValueError(f"模型流返回了不支持的 finish_reason：{finish_reason}")
            if tool_parts and finish_reason in {"length", "content_filter"}:
                raise ValueError(f"模型工具调用未完整结束：finish_reason={finish_reason}")
            if finish_reason == "tool_calls" and not tool_parts:
                raise ValueError("模型以 tool_calls 结束，但没有工具调用")

            result: dict[str, Any] = {
                "content": "".join(text_parts) if text_parts else None,
                "tool_calls": _finish_tool_calls(tool_parts),
                "finish_reason": finish_reason,
            }
            if reasoning_parts:
                result["reasoning_content"] = "".join(reasoning_parts)
            if usage is not None:
                result["usage"] = usage
            return result
        finally:
            # 也覆盖读取异常、调用方回调异常和 KeyboardInterrupt。
            response_stream.close()


def _append_tool_delta(parts: dict[int, dict[str, list[str]]], call: object) -> None:
    index = field(call, "index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("模型工具调用缺少有效的 index")
    if field(call, "type") not in {None, "function"}:
        raise ValueError("模型返回了不支持的工具调用类型")
    item = parts.setdefault(index, {"id": [], "name": [], "arguments": []})
    function = field(call, "function")
    for key, value in (
        ("id", field(call, "id")),
        ("name", field(function, "name")),
        ("arguments", field(function, "arguments")),
    ):
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"模型工具调用的 {key} 片段必须是字符串")
            item[key].append(value)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"工具参数包含非 JSON 常量：{value}")


def _finish_tool_calls(parts: dict[int, dict[str, list[str]]]) -> list[dict]:
    calls: list[dict] = []
    seen_ids: set[str] = set()
    for index in sorted(parts):
        item = {key: "".join(fragments) for key, fragments in parts[index].items()}
        if not item["id"].strip() or not item["name"].strip():
            raise ValueError("模型返回的工具调用缺少 id 或 name")
        if item["id"] in seen_ids:
            raise ValueError("模型返回了重复的工具调用 id")
        seen_ids.add(item["id"])
        try:
            arguments = json.loads(item["arguments"], parse_constant=_reject_json_constant)
        except ValueError as error:
            raise ValueError("模型工具调用参数不是完整的 JSON 对象") from error
        if not isinstance(arguments, dict):
            raise ValueError("模型工具调用参数必须是 JSON 对象")
        calls.append(
            {
                "id": item["id"],
                "type": "function",
                "function": {"name": item["name"], "arguments": item["arguments"]},
            }
        )
    return calls


def _stream_usage(usage: object) -> dict[str, Any]:
    cached = field(usage, "prompt_cache_hit_tokens")
    if cached is None:
        cached = field(field(usage, "prompt_tokens_details"), "cached_tokens")
    return {
        "prompt_tokens": field(usage, "prompt_tokens"),
        "completion_tokens": field(usage, "completion_tokens"),
        "total_tokens": field(usage, "total_tokens"),
        "cached_input_tokens": cached or 0,
        "reasoning_tokens": field(field(usage, "completion_tokens_details"), "reasoning_tokens") or 0,
    }
