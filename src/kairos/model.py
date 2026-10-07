"""DeepSeek 的 OpenAI 兼容模型适配层。

``agent_loop`` 只依赖一个很小的 ``Model.complete`` 接口；本模块负责把 OpenAI
SDK 的对象响应转换成循环能使用的普通字典。这样真实 API 和 Fake Model 可以共用
同一条工具调用路径。
"""

from collections.abc import Mapping
from typing import Any

from openai import OpenAI

from .config import Settings


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


class DeepSeekModel:
    """使用 OpenAI SDK 调用 DeepSeek，并返回循环使用的统一字典。"""

    def __init__(self, settings: Settings, client: Any | None = None):
        self.model = settings.kairos_model
        self.client = client or OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )

    def complete(self, *, messages: list[dict], tools: list[dict]) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            extra_body={"thinking": {"type": "disabled"}},
        )
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
        finish_reason = _field(choices[0], "finish_reason")
        if finish_reason is not None:
            result["finish_reason"] = finish_reason
        reasoning_content = _field(message, "reasoning_content")
        if reasoning_content is not None:
            result["reasoning_content"] = reasoning_content
        return result
