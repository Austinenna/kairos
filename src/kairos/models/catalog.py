"""已知模型的能力表。新增模型或核对能力时只改这里。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    context_window: int | None = None  # 最大上下文 token 数；None 表示未知
    thinking_toggle: bool = False  # 能否通过 extra_body.thinking 开关思考模式（DeepSeek）
    json_mode: bool = False  # 是否支持 response_format={"type": "json_object"}


MODELS: dict[str, ModelInfo] = {
    # 2026-10 实测：DeepSeek API 返回 maximum context length 1048576。
    "deepseek-flash": ModelInfo(context_window=1_048_576, thinking_toggle=True, json_mode=True),
    "deepseek-v4-pro": ModelInfo(context_window=1_000_000, thinking_toggle=True, json_mode=True),
}


def model_info(name: str) -> ModelInfo:
    """查找模型能力；不在表里的模型按“能力未知”处理。"""
    return MODELS.get(name, ModelInfo())
