"""Anthropic Messages API 适配器。"""

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
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
            if "provider_content" in message:
                blocks = message["provider_content"]
                if not isinstance(blocks, list):
                    raise ValueError("Anthropic provider_content 必须是内容块列表")
                converted.append({"role": "assistant", "content": deepcopy(blocks)})
                continue
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


def _content_block(block: object) -> dict:
    """复制原始内容块，保留回放所需的思考签名和不透明数据。"""
    if isinstance(block, Mapping):
        return deepcopy(dict(block))
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        return dump(mode="json", exclude_unset=True)
    return deepcopy(vars(block))


def normalize_anthropic_response(response: object) -> dict:
    """把 Anthropic SDK 的 Message 响应转换成 Agent Loop 使用的字典。"""

    content = field(response, "content", []) or []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict] = []
    call_ids: set[str] = set()
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
            tool_input = field(block, "input")
            if (
                not isinstance(call_id, str)
                or not call_id.strip()
                or not isinstance(name, str)
                or not name.strip()
                or not isinstance(tool_input, Mapping)
            ):
                raise ValueError("Anthropic 工具调用缺少 id、name 或对象参数")
            if call_id in call_ids:
                raise ValueError(f"Anthropic 工具调用 id 重复：{call_id}")
            call_ids.add(call_id)
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

    result = {
        "content": "\n".join(text_parts) or None,
        "tool_calls": tool_calls,
        "provider_content": [_content_block(block) for block in content],
    }
    if reasoning_parts:
        result["reasoning_content"] = "\n".join(reasoning_parts)

    usage = field(response, "usage")
    if usage is not None:
        input_tokens = field(usage, "input_tokens")
        output_tokens = field(usage, "output_tokens")
        cache_read = field(usage, "cache_read_input_tokens")
        cache_creation = field(usage, "cache_creation_input_tokens")
        # Anthropic 将普通输入、缓存读取和缓存写入分别计数。
        prompt_tokens = None if input_tokens is None else input_tokens + (cache_read or 0) + (cache_creation or 0)
        result["usage"] = {
            "prompt_tokens": prompt_tokens,
            "cached_input_tokens": cache_read,
            "completion_tokens": output_tokens,
            "total_tokens": (prompt_tokens or 0) + (output_tokens or 0),
        }
    stop_reason = field(response, "stop_reason")
    if stop_reason is not None:
        result["finish_reason"] = stop_reason
    return result


class _MessageStream:
    """从原始 SDK 事件组装一条完整消息；工具参数在块结束后才解析。"""

    def __init__(self, on_text: Callable[[str], None]):
        self.on_text = on_text
        self.started = False
        self.stopped = False
        self.blocks: dict[int, dict] = {}
        self.closed_blocks: set[int] = set()
        self.displayed_blocks: set[int] = set()
        self.arguments: dict[int, list[str]] = {}
        self.usage: dict = {}
        self.stop_reason: str | None = None

    def _usage(self, usage: object) -> None:
        # message_delta 的用量是累计值；缺失字段不能抹掉 message_start 的输入量。
        for name in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            value = field(usage, name)
            if value is not None:
                self.usage[name] = value

    def _text(self, index: int, text: object) -> None:
        if not isinstance(text, str):
            raise ValueError("Anthropic 正文片段必须是字符串")
        if not text:
            return
        if index not in self.displayed_blocks:
            if self.displayed_blocks:
                self.on_text("\n")
            self.displayed_blocks.add(index)
        self.on_text(text)

    def add(self, event: object) -> None:
        kind = field(event, "type")
        if kind == "ping":
            return
        if self.stopped:
            raise ValueError("Anthropic message_stop 后仍有消息事件")
        if kind == "message_start":
            if self.started:
                raise ValueError("Anthropic message_start 重复")
            self.started = True
            message = field(event, "message")
            if field(message, "content", []):
                raise ValueError("Anthropic message_start 包含意外的内容块")
            self._usage(field(message, "usage"))
            return
        if not self.started:
            raise ValueError("Anthropic 流缺少 message_start")
        if kind == "message_delta":
            reason = field(field(event, "delta"), "stop_reason")
            if reason is not None:
                if not isinstance(reason, str) or not reason:
                    raise ValueError("Anthropic stop_reason 无效")
                self.stop_reason = reason
            self._usage(field(event, "usage"))
            return
        if kind == "message_stop":
            self.stopped = True
            return
        if kind not in {"content_block_start", "content_block_delta", "content_block_stop"}:
            raise ValueError(f"Anthropic 返回了不支持的流事件：{kind}")
        index = field(event, "index")
        if type(index) is not int or index < 0:
            raise ValueError("Anthropic 内容块索引无效")
        if kind == "content_block_start":
            if index in self.blocks:
                raise ValueError("Anthropic 内容块索引重复")
            block = _content_block(field(event, "content_block"))
            if block.get("type") not in {"text", "thinking", "redacted_thinking", "tool_use"}:
                raise ValueError(f"Anthropic 返回了不支持的内容块：{block.get('type')}")
            self.blocks[index] = block
            if block["type"] == "text":
                self._text(index, block.get("text", ""))
            return
        if index not in self.blocks or index in self.closed_blocks:
            raise ValueError("Anthropic 内容块未开始或已经结束")
        block = self.blocks[index]
        if kind == "content_block_stop":
            if index in self.arguments:

                def reject_constant(value: str) -> None:
                    raise ValueError(f"Anthropic 工具参数包含无效 JSON 常量：{value}")

                block["input"] = json.loads("".join(self.arguments[index]), parse_constant=reject_constant)
            self.closed_blocks.add(index)
            return
        delta = field(event, "delta")
        delta_type = field(delta, "type")
        fields = {
            "text_delta": ("text", "text"),
            "thinking_delta": ("thinking", "thinking"),
            "signature_delta": ("thinking", "signature"),
            "input_json_delta": ("tool_use", "partial_json"),
        }
        if delta_type not in fields or block["type"] != fields[delta_type][0]:
            raise ValueError(f"Anthropic 内容块与增量类型不匹配：{delta_type}")
        _, name = fields[delta_type]
        value = field(delta, name)
        if not isinstance(value, str):
            raise ValueError("Anthropic 内容增量必须是字符串")
        if delta_type == "input_json_delta":
            self.arguments.setdefault(index, []).append(value)
        else:
            block[name] = block.get(name, "") + value
            if delta_type == "text_delta":
                self._text(index, value)

    def result(self) -> dict:
        if not self.stopped or not self.stop_reason or self.closed_blocks != self.blocks.keys():
            raise ValueError("Anthropic 流未完整结束：需要 message_stop、stop_reason 和所有内容块结束")
        result = normalize_anthropic_response(
            {
                "content": [self.blocks[index] for index in sorted(self.blocks)],
                "stop_reason": self.stop_reason,
                "usage": self.usage or None,
            }
        )
        if bool(result["tool_calls"]) != (self.stop_reason == "tool_use"):
            raise ValueError("Anthropic 工具调用与 tool_use 结束原因不一致")
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
        response = self.client.messages.create(**self._request(messages, tools))
        return normalize_anthropic_response(response)

    def stream(self, *, messages: list[dict], tools: list[dict], on_text: Callable[[str], None]) -> dict:
        """即时发出正文，收齐并校验消息后返回可执行的工具调用。"""
        response = self.client.messages.create(**self._request(messages, tools), stream=True)
        try:
            message = _MessageStream(on_text)
            for event in response:
                message.add(event)
            return message.result()
        finally:
            response.close()

    def _request(self, messages: list[dict], tools: list[dict]) -> dict:
        system, claude_messages = anthropic_messages(messages)
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": claude_messages,
            "tools": anthropic_tools(tools),
        }
        if system is not None:
            request["system"] = system
        return request
