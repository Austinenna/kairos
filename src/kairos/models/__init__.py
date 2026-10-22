"""模型层：把不同供应商的协议统一成 Model.stream() -> ModelResponse。"""

from ..config import Settings
from .anthropic import AnthropicModel
from .base import Model, ModelResponse, Usage
from .catalog import model_info
from .openai_compat import OpenAICompatibleModel

__all__ = ["Model", "ModelResponse", "Usage", "build_model"]


def build_model(settings: Settings, *, thinking: bool = False) -> Model:
    """根据配置创建模型适配器。

    thinking 只对能力表里支持思考开关的模型生效。
    """
    info = model_info(settings.kairos_model)
    if settings.kairos_provider == "anthropic":
        return AnthropicModel(
            settings.kairos_model,
            api_key=settings.kairos_api_key,
            base_url=settings.anthropic_base_url or _strip_v1(settings.gateway_base_url),
            max_tokens=settings.anthropic_max_tokens,
        )
    return OpenAICompatibleModel(
        settings.kairos_model,
        api_key=settings.kairos_api_key,
        base_url=settings.gateway_base_url,
        thinking=thinking if info.thinking_toggle else None,
    )


def _strip_v1(url: str) -> str:
    """Anthropic SDK 会自己拼 /v1/messages，所以网关地址末尾的 /v1 要去掉。"""
    url = url.rstrip("/")
    return url.removesuffix("/v1")
