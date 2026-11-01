"""回看一轮：把数据库里这一轮的消息排成“请求 → 工具 → 回答”的步骤，给 /trace 显示。

不另建表，数据都来自 messages：助手消息带着工具调用和这次请求的用量，工具结果（包括报错）是 tool 消息。
耗时由相邻两条消息的保存时间相减得出：请求耗时 = 助手消息的保存时间 − 上一条消息的保存时间，工具同理。
2026-11-01 之前的消息时间只精确到秒，算出的耗时不准。
"""

import itertools
import json
from dataclasses import dataclass

from ..agent.loop import INTERRUPTED_NOTE
from ..images.prepare import text_of
from ..storage import MessageRecord, SessionStore

TERMINAL_WIDTH = 160  # 参数、结果和回答的预览最多显示多少字
WEIXIN_WIDTH = 40


@dataclass
class TraceLine:
    text: str
    error: bool = False  # 工具报错的那一行，终端显示成红色


def load_trace(sessions: SessionStore, session_id: str, argument: str, width: int) -> list[TraceLine]:
    """处理 /trace 命令：argument 里的数字是倒数第几轮（默认 1），其余当作会话 ID（默认当前会话）。

    会话 ID 可以只写开头几位，但至少 4 位，免得和轮数混淆；找不到时抛出 ValueError。
    """
    back, reference = 1, None
    for word in argument.split():
        if word.isdigit() and len(word) <= 3:
            back = max(int(word), 1)
        else:
            reference = word
    if reference:
        session_id = sessions.get(reference).id
    return trace_lines(sessions.turn(session_id, back), width)


def trace_lines(records: list[MessageRecord], width: int = TERMINAL_WIDTH) -> list[TraceLine]:
    """把一轮的消息排成显示用的行。records[0] 是这一轮的用户消息。"""
    if not records:
        return [TraceLine("没有这一轮：会话里还没有这么多条用户消息。")]
    lines = [TraceLine(_summary(records)), TraceLine(f"用户：{_clip(text_of(records[0].message['content']), width)}")]
    calls: dict[str, dict] = {}  # tool_call_id → 调用，工具结果按 id 找回自己的调用
    requests = 0
    for previous, record in itertools.pairwise(records):
        message, seconds = record.message, (record.created_at - previous.created_at).total_seconds()
        content = text_of(message.get("content") or "")
        if message["role"] == "tool":
            function = calls[message["tool_call_id"]]
            lines.append(TraceLine(f"  → {function['name']} {_clip(function['arguments'], width)}  {seconds:.1f}s"))
            lines.append(_result_line(content, width))
        elif content.endswith(INTERRUPTED_NOTE):
            requests += 1
            lines.append(TraceLine(f"第 {requests} 次请求（被中断，没有用量）  {seconds:.1f}s"))
            lines.append(TraceLine(f"  已显示：{_clip(content.removesuffix(INTERRUPTED_NOTE), width)}"))
        elif record.usage is None:  # 不是模型生成的，例如定时推送追加进会话的消息
            lines.append(TraceLine(f"程序写入：{_clip(content, width)}"))
        else:
            requests += 1
            lines.append(TraceLine(f"第 {requests} 次请求  {_usage(record)}  {seconds:.1f}s"))
            for call in message.get("tool_calls", []):
                calls[call["id"]] = call["function"]
            if content:
                label = "说明" if message.get("tool_calls") else "回答"
                lines.append(TraceLine(f"  {label}：{_clip(content, width)}"))
    return lines


def _summary(records: list[MessageRecord]) -> str:
    """第一行：开始时间、请求次数、合计用量和费用、总用时。"""
    requests = [record for record in records if record.usage]
    total = (records[-1].created_at - records[0].created_at).total_seconds()
    usage = (
        f"输入 {_k(sum(r.usage.input_tokens or 0 for r in requests))}"
        f"（缓存 {_k(sum(r.usage.cached_input_tokens or 0 for r in requests))}）"
        f"输出 {_k(sum(r.usage.output_tokens or 0 for r in requests))}"
    )
    cost = sum(record.cost_usd or 0 for record in requests)
    return f"{records[0].created_at:%m-%d %H:%M:%S} · {len(requests)} 次请求 · {usage} · ${cost:.4f} · 共 {total:.1f}s"


def _usage(record: MessageRecord) -> str:
    usage = record.usage
    text = f"输入 {_k(usage.input_tokens)}（缓存 {_k(usage.cached_input_tokens)}）输出 {_k(usage.output_tokens)}"
    return text + (f"  ${record.cost_usd:.4f}" if record.cost_usd is not None else "")


def _result_line(content: str, width: int) -> TraceLine:
    """工具结果：报错的显示错误信息并标红，其余显示长度和开头一段。"""
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = None
    if isinstance(result, dict) and "error" in result:
        return TraceLine(f"    ✗ {_clip(str(result['error']), width)}", error=True)
    return TraceLine(f"    ← {len(content):,} 字：{_clip(content, width)}")


def _clip(text: str, width: int) -> str:
    """压成一行，超过 width 个字时截断。"""
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width] + "…"


def _k(n: int | None) -> str:
    """token 数缩写：12345 → 12.3K；供应商没给时显示 ?。"""
    if n is None:
        return "?"
    return f"{n / 1000:.1f}K" if n >= 1000 else str(n)
