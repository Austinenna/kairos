"""模型层的公共类型：所有供应商适配器都返回同样的 ModelResponse。

内部消息统一使用 OpenAI Chat Completions 的形状：
    {"role": "system" | "user" | "assistant" | "tool", "content": ..., ...}
助手消息可带 tool_calls，工具结果消息带 tool_call_id。
各家协议的差异只在 models/ 里的适配器中处理。
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Usage:
    """一次模型请求的 token 用量；供应商没给的字段为 None。"""

    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass
class ModelResponse:
    """一次模型请求的完整结果。"""

    content: str
    # OpenAI 风格：{"id", "type": "function", "function": {"name", "arguments": JSON 字符串}}
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage | None = None
    # 下次请求必须原样回传给同一模型的字段，例如 DeepSeek 的 reasoning_content、
    # Anthropic 带思考签名的 provider_content。
    extra: dict = field(default_factory=dict)

    def to_message(self) -> dict:
        """转换成可以追加到历史里的助手消息。"""
        message = {"role": "assistant", "content": self.content, **self.extra}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return message


class Model(Protocol):
    """Agent 循环需要的唯一接口：流式请求一次模型。"""

    model: str

    def stream(
        self, messages: list[dict], tools: list[dict], on_text: Callable[[str], None] | None = None
    ) -> ModelResponse: ...
