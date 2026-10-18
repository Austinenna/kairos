"""token 用量和费用估算。

价格来自 DeepSeek 官方定价页（2026-10 查询），单位是美元 / 每百万 token。
价格会调整，用之前到 https://api-docs.deepseek.com/quick_start/pricing 核对。
"""

from dataclasses import dataclass
from datetime import UTC, datetime

from .providers.openai_compatible import normalize_openai_usage

# (输入·缓存命中, 输入·缓存未命中, 输出)
PRICES: dict[str, dict[str, tuple[float, float, float]]] = {
    "deepseek-flash": {"off_peak": (0.003, 0.15, 0.60), "peak": (0.006, 0.30, 1.20)},
    "deepseek-v4-pro": {"off_peak": (0.022, 0.66, 1.98), "peak": (0.044, 1.32, 3.96)},
}

CONTEXT_WINDOW: dict[str, int] = {
    # 2026-10 实测：DeepSeek API 返回 maximum context length 1048576。
    "deepseek-flash": 1_048_576,
    # 该模型尚未在本项目中做同样的超限实测，暂保留原值。
    "deepseek-v4-pro": 1_000_000,
}

# 高峰时段（UTC，周一到周五），价格翻倍。
# 换算成北京时间是 9:00–12:00 和 14:00–18:00，正好是工作时间。
PEAK_HOURS_UTC = [(1, 4), (6, 10)]


@dataclass
class Usage:
    input_tokens: int | None
    cached_input_tokens: int | None  # 输入里命中缓存的部分；None 表示供应商没有给出有效计数
    output_tokens: int | None  # 包含思考用掉的 token
    reasoning_tokens: int | None


def usage_from_openai(u: object) -> Usage:
    """通过统一协议转换将 OpenAI 用量映射为 Usage。

    作用:
        保留旧用量入口，与模型适配器使用相同字段来源和未知值处理规则。

    参数:
        u: OpenAI 兼容的用量对象或字典。

    返回:
        结构化用量（Usage）；未提供或无效的计数字段为 None。

    行为说明:
        1. 委托 normalize_openai_usage 解析字段，不重复实现缓存来源选择。
        2. 将统一字段映射为费用和展示使用的 Usage，不计算任何累计值。
    """

    usage = normalize_openai_usage(u)
    return Usage(
        input_tokens=usage["prompt_tokens"],
        cached_input_tokens=usage["cached_input_tokens"],
        output_tokens=usage["completion_tokens"],
        reasoning_tokens=usage["reasoning_tokens"],
    )


def is_peak(now: datetime | None = None) -> bool:
    now = now or datetime.now(UTC)
    if now.weekday() >= 5:
        return False
    return any(start <= now.hour < end for start, end in PEAK_HOURS_UTC)


def estimate_cost(model: str, usage: Usage, now: datetime | None = None) -> float | None:
    """按已知用量和现有价格表估算费用。

    作用:
        为终端展示计算美元费用；关键计数未知时不输出误导性的估算。

    参数:
        model: 价格表中的模型名称。
        usage: 请求或回合累计用量，输入、输出和缓存输入计数可为 None。
        now: 判断高峰价格的 UTC 时间；省略时使用当前时间。

    返回:
        估算费用（float，美元）；模型价格未知或关键计数缺失时返回 None。

    行为说明:
        1. 确认模型价格和输入、输出、缓存计数均已知。
        2. 输入按缓存命中与未命中分别计费，输出整体计费。
        3. 思考 token 已包含在输出量中，思考明细缺失不妨碍费用估算。
    """

    table = PRICES.get(model)
    if table is None or usage.input_tokens is None or usage.output_tokens is None or usage.cached_input_tokens is None:
        return None
    hit, miss, out = table["peak" if is_peak(now) else "off_peak"]
    uncached = usage.input_tokens - usage.cached_input_tokens
    return (usage.cached_input_tokens * hit + uncached * miss + usage.output_tokens * out) / 1_000_000
