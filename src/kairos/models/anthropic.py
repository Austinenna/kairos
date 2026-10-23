"""Anthropic Messages 协议的适配器。

内部历史是 OpenAI 形状，这里负责双向转换：
- 请求：system 单独传；工具结果合并进一条 user 消息；工具定义改成 input_schema。
- 响应：tool_use 块转成 OpenAI 风格的 tool_calls；原始内容块存进 provider_content，
  下次请求原样回传，保留思考签名和块的顺序。
"""

import json
from collections.abc import Callable

from anthropic import Anthropic

from .base import ModelResponse, Usage
from .openai_compat import REQUEST_TIMEOUT


class AnthropicModel:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        client: Anthropic | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.client = client or Anthropic(api_key=api_key, base_url=base_url, timeout=REQUEST_TIMEOUT)

    def stream(
        self, messages: list[dict], tools: list[dict], on_text: Callable[[str], None] | None = None
    ) -> ModelResponse:
        """发起一次流式请求；SDK 的 stream 助手负责把事件拼成完整消息。"""
        system, converted = to_anthropic_messages(messages)
        request = {"model": self.model, "max_tokens": self.max_tokens, "messages": converted}
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [to_anthropic_tool(tool) for tool in tools]

        with self.client.messages.stream(**request) as stream:
            for text in stream.text_stream:
                if on_text:
                    on_text(text)
            message = stream.get_final_message()

        text_parts = []
        tool_calls = []
        for block in message.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {"name": block.name, "arguments": json.dumps(block.input, ensure_ascii=False)},
                    }
                )
        return ModelResponse(
            content="".join(text_parts),
            tool_calls=tool_calls,
            finish_reason=message.stop_reason,
            usage=_usage(message.usage),
            extra={"provider_content": [block.model_dump(mode="json", exclude_none=True) for block in message.content]},
        )


def to_anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """把内部历史转换成 (system 文本, Anthropic 消息列表)。"""
    system_parts = []
    converted = []
    for message in messages:
        role = message["role"]
        if role == "system":
            system_parts.append(message["content"])
        elif role == "tool":
            # 工具结果可以带图片（view_image），和用户消息里的图片一样转换。
            content = _user_content(message["content"])
            result = {"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": content}
            # 同一批工具的结果必须放在同一条 user 消息里。
            if converted and converted[-1]["role"] == "user" and _is_tool_results(converted[-1]):
                converted[-1]["content"].append(result)
            else:
                converted.append({"role": "user", "content": [result]})
        elif role == "assistant" and "provider_content" in message:
            converted.append({"role": "assistant", "content": message["provider_content"]})
        elif role == "assistant":
            blocks = [{"type": "text", "text": message["content"]}] if message.get("content") else []
            for call in message.get("tool_calls", []):
                function = call["function"]
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": function["name"],
                        "input": json.loads(function["arguments"]),
                    }
                )
            converted.append({"role": "assistant", "content": blocks})
        else:
            converted.append({"role": "user", "content": _user_content(message["content"])})
    return "\n\n".join(system_parts), converted


def _user_content(content: str | list[dict]) -> str | list[dict]:
    """图片块转成 Anthropic 的 base64 image 块。用户消息和工具结果都用它。"""
    if isinstance(content, str):
        return content
    return [
        {"type": "image", "source": {"type": "base64", "media_type": block["mime_type"], "data": block["data"]}}
        if block["type"] == "image"
        else block
        for block in content
    ]


def to_anthropic_tool(tool: dict) -> dict:
    """OpenAI 工具定义 → Anthropic 工具定义。"""
    function = tool["function"]
    return {
        "name": function["name"],
        "description": function.get("description", ""),
        "input_schema": function["parameters"],
    }


def _is_tool_results(message: dict) -> bool:
    content = message["content"]
    return isinstance(content, list) and all(block.get("type") == "tool_result" for block in content)


def _usage(raw) -> Usage:
    # Anthropic 的 input_tokens 不含缓存读写部分，三者相加才是完整输入。
    cache_read = raw.cache_read_input_tokens or 0
    cache_write = raw.cache_creation_input_tokens or 0
    return Usage(
        input_tokens=raw.input_tokens + cache_read + cache_write,
        cached_input_tokens=cache_read,
        output_tokens=raw.output_tokens,
    )
