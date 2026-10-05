"""token 用量和费用估算。

价格来自 DeepSeek 官方定价页（2026-10 查询），单位是美元 / 每百万 token。
价格会调整，用之前到 https://api-docs.deepseek.com/quick_start/pricing 核对。
"""

from dataclasses import dataclass
from datetime import UTC, datetime

# (输入·缓存命中, 输入·缓存未命中, 输出)
PRICES: dict[str, dict[str, tuple[float, float, float]]] = {
    "deepseek-flash": {"off_peak": (0.003, 0.15, 0.60), "peak": (0.006, 0.30, 1.20)},
    "deepseek-v4-pro": {"off_peak": (0.022, 0.66, 1.98), "peak": (0.044, 1.32, 3.96)},
}

CONTEXT_WINDOW: dict[str, int] = {
    "deepseek-flash": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
}

# 高峰时段（UTC，周一到周五），价格翻倍。
# 换算成北京时间是 9:00–12:00 和 14:00–18:00，正好是工作时间。
PEAK_HOURS_UTC = [(1, 4), (6, 10)]


@dataclass
class Usage:
    input_tokens: int
    cached_input_tokens: int  # 输入里命中缓存的部分，价格便宜得多
    output_tokens: int  # 包含思考用掉的 token
    reasoning_tokens: int


def usage_from_openai(u) -> Usage:
    """把 OpenAI 格式的 usage 对象转成我们自己的 Usage。

    缓存命中数在不同服务商、不同版本里放的位置不一样，所以几个地方都找一下。
    """
    details = getattr(u, "prompt_tokens_details", None)
    cached = (
        getattr(u, "prompt_cache_hit_tokens", None)
        or getattr(details, "prompt_cache_hit_tokens", None)
        or getattr(details, "cached_tokens", None)
        or 0
    )
    completion_details = getattr(u, "completion_tokens_details", None)
    reasoning = getattr(completion_details, "reasoning_tokens", None) or 0
    return Usage(
        input_tokens=u.prompt_tokens,
        cached_input_tokens=cached,
        output_tokens=u.completion_tokens,
        reasoning_tokens=reasoning,
    )


def is_peak(now: datetime | None = None) -> bool:
    now = now or datetime.now(UTC)
    if now.weekday() >= 5:
        return False
    return any(start <= now.hour < end for start, end in PEAK_HOURS_UTC)


def estimate_cost(model: str, usage: Usage, now: datetime | None = None) -> float | None:
    """估算一次请求的费用（美元）。不认识的模型返回 None。"""
    table = PRICES.get(model)
    if table is None:
        return None
    hit, miss, out = table["peak" if is_peak(now) else "off_peak"]
    uncached = usage.input_tokens - usage.cached_input_tokens
    return (usage.cached_input_tokens * hit + uncached * miss + usage.output_tokens * out) / 1_000_000
