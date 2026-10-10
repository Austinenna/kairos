"""模型供应商适配器。"""

from .anthropic_messages import AnthropicMessagesModel
from .openai_compatible import OpenAICompatibleModel

__all__ = ["AnthropicMessagesModel", "OpenAICompatibleModel"]
