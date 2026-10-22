"""OpenAI Chat Completions 兼容协议的适配器（DeepSeek 默认走这里）。"""

from collections.abc import Callable

from openai import OpenAI

from .base import ModelResponse, Usage

# 流式请求的读超时（秒）。不设置时 SDK 默认等很久，Ctrl+C 之外没法停下来。
REQUEST_TIMEOUT = 120.0


class OpenAICompatibleModel:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        thinking: bool | None = None,
        json_mode: bool = False,
        client: OpenAI | None = None,
    ):
        """创建适配器。

        thinking 为 None 表示不发送思考开关（模型不支持时）；
        json_mode 为 True 时要求模型只输出 JSON 对象，后台记忆提取会用到。
        """
        self.model = model
        self.client = client or OpenAI(api_key=api_key, base_url=base_url, timeout=REQUEST_TIMEOUT)
        self.extra_body = {}
        if thinking is not None:
            self.extra_body["thinking"] = {"type": "enabled" if thinking else "disabled"}
        self.json_mode = json_mode

    def stream(
        self, messages: list[dict], tools: list[dict], on_text: Callable[[str], None] | None = None
    ) -> ModelResponse:
        """发起一次流式请求：正文片段边收边交给 on_text，结束后返回完整结果。"""
        request = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},  # 让最后一帧带上用量
        }
        if tools:
            request["tools"] = tools
        if self.extra_body:
            request["extra_body"] = self.extra_body
        if self.json_mode:
            request["response_format"] = {"type": "json_object"}

        text: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict] = {}  # 工具调用按 index 分片到达，先按 index 拼起来
        finish_reason = None
        usage = None

        with self.client.chat.completions.create(**request) as stream:
            for chunk in stream:
                if chunk.usage:
                    usage = _usage(chunk.usage)
                if not chunk.choices:  # 最后的用量帧没有 choices
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta.content:
                    text.append(delta.content)
                    if on_text:
                        on_text(delta.content)
                # DeepSeek 的思考内容，不展示，但带工具的会话里下次必须回传。
                if piece := getattr(delta, "reasoning_content", None):
                    reasoning.append(piece)
                for call in delta.tool_calls or []:
                    item = calls.setdefault(
                        call.index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                    )
                    if call.id:
                        item["id"] = call.id
                    if call.function and call.function.name:
                        item["function"]["name"] = call.function.name
                    if call.function and call.function.arguments:
                        item["function"]["arguments"] += call.function.arguments
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

        extra = {"reasoning_content": "".join(reasoning)} if reasoning else {}
        return ModelResponse(
            content="".join(text),
            tool_calls=[calls[index] for index in sorted(calls)],
            finish_reason=finish_reason,
            usage=usage,
            extra=extra,
        )


def _usage(raw) -> Usage:
    """把 SDK 用量转换成 Usage；缓存命中字段各家位置不同。"""
    cached = getattr(raw, "prompt_cache_hit_tokens", None)  # DeepSeek 放在顶层
    details = getattr(raw, "prompt_tokens_details", None)
    if cached is None and details is not None:
        cached = details.cached_tokens  # OpenAI 放在 prompt_tokens_details 里
    output_details = getattr(raw, "completion_tokens_details", None)
    return Usage(
        input_tokens=raw.prompt_tokens,
        cached_input_tokens=cached,
        output_tokens=raw.completion_tokens,
        reasoning_tokens=output_details.reasoning_tokens if output_details else None,
    )
