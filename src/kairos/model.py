"""模型适配器的兼容导出。

具体协议实现位于 ``kairos.providers``。保留本模块是为了兼容阶段 1 期间已经
存在的 ``from kairos.model import ...`` 导入路径；新的代码应直接从
``kairos.providers`` 导入适配器。
"""

from .providers.anthropic_messages import (
    AnthropicMessagesModel,
    anthropic_messages,
    anthropic_tools,
    normalize_anthropic_response,
)
from .providers.common import field, normalize_tool_call
from .providers.openai_compatible import OpenAICompatibleModel

__all__ = [
    "AnthropicMessagesModel",
    "OpenAICompatibleModel",
    "anthropic_messages",
    "anthropic_tools",
    "field",
    "normalize_anthropic_response",
    "normalize_tool_call",
]
