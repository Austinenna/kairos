"""已知模型的能力和价格。新增模型、核对能力或调价时只改这里。

价格来自 DeepSeek 官方定价页（2026-10 查询），单位是美元 / 每百万 token。
价格会调整，改动前到 https://api-docs.deepseek.com/quick_start/pricing 核对。
"""

from dataclasses import dataclass
from datetime import UTC, datetime

from .base import Usage

# (输入·缓存命中, 输入·缓存未命中, 输出)，单位：美元 / 百万 token
Prices = tuple[float, float, float]


@dataclass(frozen=True)
class ModelInfo:
    context_window: int | None = None  # 最大上下文 token 数；None 表示未知
    thinking_toggle: bool = False  # 能否通过 extra_body.thinking 开关思考模式（DeepSeek）
    off_peak_prices: Prices | None = None  # None 表示不知道价格，不估算费用
    peak_prices: Prices | None = None


MODELS: dict[str, ModelInfo] = {
    # 2026-10 实测：DeepSeek API 返回 maximum context length 1048576。
    "deepseek-flash": ModelInfo(
        context_window=1_048_576,
        thinking_toggle=True,
        off_peak_prices=(0.003, 0.15, 0.60),
        peak_prices=(0.006, 0.30, 1.20),
    ),
    "deepseek-v4-pro": ModelInfo(
        context_window=1_000_000,
        thinking_toggle=True,
        off_peak_prices=(0.022, 0.66, 1.98),
        peak_prices=(0.044, 1.32, 3.96),
    ),
}

# DeepSeek 高峰时段（UTC，周一到周五），价格翻倍。即北京时间 9:00–12:00 和 14:00–18:00。
PEAK_HOURS_UTC = [(1, 4), (6, 10)]


def model_info(name: str) -> ModelInfo:
    """查找模型能力；不在表里的模型按“能力未知”处理。"""
    return MODELS.get(name, ModelInfo())


def estimate_cost(model: str, usage: Usage, at: datetime | None = None) -> float | None:
    """估算一次请求的费用（美元）。价格未知或用量缺项时返回 None。"""
    info = model_info(model)
    at = (at or datetime.now(UTC)).astimezone(UTC)
    peak = at.weekday() < 5 and any(start <= at.hour < end for start, end in PEAK_HOURS_UTC)
    prices = info.peak_prices if peak else info.off_peak_prices
    if prices is None or usage.input_tokens is None or usage.output_tokens is None:
        return None
    hit, miss, out = prices
    cached = usage.cached_input_tokens or 0
    return (cached * hit + (usage.input_tokens - cached) * miss + usage.output_tokens * out) / 1_000_000
