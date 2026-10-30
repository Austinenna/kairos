"""压缩流程：拼出发给模型的历史、估算大小、选切点、写摘要并保存。

数据库里的原文不删不改。压缩只往 compactions 表加一行，之后发给模型的历史变成：
    [system 提示词] [摘要（user）] [从 first_kept_message_id 开始的原文]
"""

from collections.abc import Iterable
from dataclasses import dataclass

from ..models import Model
from ..storage import Compaction, SessionStore, StoredMessage
from .budget import estimate_tokens
from .summary import summarize, summary_message

KEEP_RECENT_TOKENS = 20_000  # 压缩后保留多少原文（估算值），照 pi 的 keepRecentTokens


@dataclass
class CompactResult:
    tokens_before: int
    tokens_after: int  # 估算值，不含请求时才加上的规则、记忆和工具定义


def load_context(sessions: SessionStore, session_id: str) -> list[dict]:
    """发给模型的历史。没压缩过就是完整历史。"""
    compaction, rows = _load(sessions, session_id)
    return _context(compaction, rows, session_id)


def context_tokens(sessions: SessionStore, session_id: str, extra: Iterable[dict] = ()) -> int:
    """当前上下文大小：最近一次压缩之后、最近一条有用量的助手消息的 input + output，加上它之后所有消息的估算值。

    压缩之前的助手消息用量反映的是压缩前的大小，不能用；压缩后还没请求过模型时，整段按字符估算。
    extra 是还没存进数据库的消息，例如这一轮刚输入的用户消息。
    """
    compaction, rows = _load(sessions, session_id)
    after = compaction.last_message_id if compaction else 0
    for index in range(len(rows) - 1, -1, -1):
        if rows[index].id > after and rows[index].context_tokens is not None:
            newer = [row.message for row in rows[index + 1 :]]
            return rows[index].context_tokens + estimate_tokens([*newer, *extra])
    return estimate_tokens([*_context(compaction, rows, session_id), *extra])


def compact(
    sessions: SessionStore, session_id: str, model: Model, tokens_before: int, focus: str = ""
) -> CompactResult | None:
    """把较早的消息（连同上一次的摘要）写成一份新摘要并保存。没有可以压缩的消息时返回 None。

    写摘要时被 Ctrl+C 中断，什么都不保存，历史保持原样。
    """
    compaction, rows = _load(sessions, session_id)
    cut = split_point(rows)
    if cut is None:
        return None
    previous = compaction.summary if compaction else None
    summary = summarize(model, [row.message for row in rows[1:cut]], previous, focus)
    sessions.add_compaction(session_id, summary, rows[cut].id, rows[-1].id, tokens_before)
    return CompactResult(tokens_before, estimate_tokens(load_context(sessions, session_id)))


def split_point(rows: list[StoredMessage]) -> int | None:
    """选出保留原文的第一条消息（下标）。

    从最新的消息往回累计到 KEEP_RECENT_TOKENS，切在用户消息上，所以工具调用和结果不会被拆开，
    至少保留最近一整轮。rows[0] 是 system 提示词；前面没有可以压缩的消息时返回 None。
    """
    kept = 0
    for index in range(len(rows) - 1, 0, -1):
        kept += estimate_tokens([rows[index].message])
        if rows[index].message["role"] == "user" and kept >= KEEP_RECENT_TOKENS:
            return index if index > 1 else None
    return None


def _load(sessions: SessionStore, session_id: str) -> tuple[Compaction | None, list[StoredMessage]]:
    """最新的压缩记录，和 system 提示词加上它保留的原文。"""
    compaction = sessions.latest_compaction(session_id)
    return compaction, sessions.load_rows(session_id, compaction.first_kept_message_id if compaction else 0)


def _context(compaction: Compaction | None, rows: list[StoredMessage], session_id: str) -> list[dict]:
    messages = [row.message for row in rows]
    if compaction is None:
        return messages
    return [messages[0], summary_message(compaction.summary, session_id), *messages[1:]]
