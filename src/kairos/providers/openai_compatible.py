"""OpenAI Chat Completions 兼容模型适配器。"""

from typing import Any

from openai import OpenAI

from .common import field, normalize_tool_call


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
        request = {"model": self.model, "messages": messages, "tools": tools}
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
