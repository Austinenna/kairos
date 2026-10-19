"""从已保存的模型请求明细汇总会话用量，不重复保存累计值。"""

from dataclasses import dataclass
from math import fsum, isfinite

from .pricing import Usage, usage_from_normalized


@dataclass(frozen=True)
class SessionUsage:
    """当前会话的累计用量、已估算费用及最后一次请求输入。"""

    usage: Usage
    cost: float
    step_count: int
    unpriced_steps: int
    legacy_estimated_steps: int
    last_request_usage: Usage | None


def summarize_usage(steps: list[dict]) -> SessionUsage:
    """按步骤明细汇总会话统计，保留未知字段和未估算费用的数量。

    作用:
        将存储层返回的请求明细转换为终端需要的会话统计；只做计算，不访问数据库。

    参数:
        steps: 按执行顺序排列的步骤字典，包含已解析的 usage 和 cost_snapshot。

    返回:
        会话用量汇总（SessionUsage）；没有步骤时用量和费用均为零。

    行为说明:
        1. 逐字段相加；任一步缺少该字段则总量未知，不将缺失计数当成零。
        2. 只累加已保存的有效费用，另计未估算步骤；不根据当前时间或价格重新估算。
        3. 上下文只使用最后一步的输入，不借用前一步，也不使用会话累计输入。
    """
    usages = [usage_from_normalized(step.get("usage")) for step in steps]
    totals = {}
    for name in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"):
        values = [getattr(usage, name, None) for usage in usages]
        totals[name] = sum(values) if all(value is not None for value in values) else None

    costs = []
    unpriced = 0
    legacy = 0
    for step in steps:
        snapshot = step.get("cost_snapshot")
        amount = snapshot.get("cost_usd") if isinstance(snapshot, dict) else None
        if type(amount) in (int, float) and isfinite(amount) and amount >= 0:
            costs.append(amount)
            legacy += snapshot.get("source") == "legacy_estimate"
        else:
            unpriced += 1
    return SessionUsage(
        usage=Usage(**totals),
        cost=fsum(costs),
        step_count=len(steps),
        unpriced_steps=unpriced,
        legacy_estimated_steps=legacy,
        last_request_usage=usages[-1] if usages else None,
    )
