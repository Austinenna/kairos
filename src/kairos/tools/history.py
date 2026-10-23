"""历史会话工具：让模型搜索和阅读过去的对话。只读，不切换当前会话。"""

import json
import sqlite3

from ..images import text_of
from .registry import Toolset, tool_schema

SEARCH_HISTORY = tool_schema(
    "search_history",
    "在过去所有会话的用户和助手消息里搜索关键词，返回最近的匹配。",
    {
        "query": {"type": "string", "description": "关键词，多个用空格分开，需同时包含"},
        "limit": {"type": "integer", "description": "最多返回几条，默认 5"},
    },
    required=["query"],
)

READ_SESSION = tool_schema(
    "read_session",
    "按会话 ID 阅读一个历史会话的用户和助手消息，可以用 offset 往后翻页。",
    {
        "session_id": {"type": "string", "description": "search_history 返回的会话 ID"},
        "offset": {"type": "integer", "description": "从第几条消息开始，默认 0"},
    },
    required=["session_id"],
)

LIST_SESSIONS = tool_schema(
    "list_sessions",
    "按时间列出会话，最近活动的在前。可以只列某一天有过对话的会话。",
    {
        "date": {"type": "string", "description": "只列这一天（YYYY-MM-DD）有消息的会话，不填则列最近的"},
        "limit": {"type": "integer", "description": "最多返回几个，默认 10"},
    },
    required=[],
)

RULES = """历史会话来自终端和微信两个入口，共用同一份记录。
channel 字段表示会话在哪个入口创建：terminal 是终端，weixin 是微信。
用户问起以前讨论过的某件事时，先用 search_history 找到相关会话；
问某天或最近聊过什么时，用 list_sessions 列出会话。找到后用 read_session 阅读上下文。"""

PAGE_SIZE = 20
SNIPPET_CHARS = 200


def history_tools(db: sqlite3.Connection) -> Toolset:
    def search_history(query: str, limit: int = 5) -> list[dict]:
        words = query.split()
        # 每个关键词一个 LIKE 条件，全部满足才算匹配。content 从消息 JSON 里取。
        conditions = " AND ".join("json_extract(m.message, '$.content') LIKE ?" for _ in words)
        rows = db.execute(
            f"""SELECT m.session_id, s.title, s.channel, m.role, m.created_at,
                       json_extract(m.message, '$.content') AS content, json_type(m.message, '$.content') AS kind
                FROM messages m JOIN sessions s ON s.id = m.session_id
                WHERE m.role IN ('user', 'assistant') AND {conditions}
                ORDER BY m.id DESC LIMIT ?""",
            [*(f"%{word}%" for word in words), limit],
        ).fetchall()
        return [
            {
                "session_id": row["session_id"],
                "title": row["title"],
                "channel": row["channel"],
                "role": row["role"],
                "time": row["created_at"],
                "snippet": _snippet(_text(row), words[0]),
            }
            for row in rows
        ]

    def list_sessions(date: str = "", limit: int = 10) -> list[dict]:
        # created_at 存的是本地时间，前 10 个字符就是本地日期。
        rows = db.execute(
            """SELECT s.id, s.title, s.channel, s.created_at, s.updated_at,
                      SUM(m.role IN ('user', 'assistant')) AS messages
               FROM sessions s JOIN messages m ON m.session_id = s.id
               GROUP BY s.id
               HAVING SUM(m.role = 'user') > 0 AND (? = '' OR SUM(substr(m.created_at, 1, 10) = ?) > 0)
               ORDER BY s.updated_at DESC LIMIT ?""",
            (date, date, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def read_session(session_id: str, offset: int = 0) -> dict:
        rows = db.execute(
            """SELECT role, created_at, json_extract(message, '$.content') AS content,
                      json_type(message, '$.content') AS kind
               FROM messages
               WHERE session_id = ? AND role IN ('user', 'assistant') AND content != ''
               ORDER BY id LIMIT ? OFFSET ?""",
            (session_id, PAGE_SIZE + 1, offset),
        ).fetchall()
        messages = [{"role": row["role"], "time": row["created_at"], "content": _text(row)} for row in rows]
        has_more = len(messages) > PAGE_SIZE
        return {
            "messages": messages[:PAGE_SIZE],
            "next_offset": offset + PAGE_SIZE if has_more else None,
        }

    return Toolset(
        schemas=[SEARCH_HISTORY, LIST_SESSIONS, READ_SESSION],
        handlers={"search_history": search_history, "list_sessions": list_sessions, "read_session": read_session},
        instructions=[RULES],
    )


def _text(row: sqlite3.Row) -> str:
    """纯文字消息原样返回；图文消息的 content 是 JSON 数组，转成不带 base64 的文字。"""
    return text_of(json.loads(row["content"])) if row["kind"] == "array" else row["content"]


def _snippet(content: str, word: str) -> str:
    """截取关键词附近的一段文字。"""
    start = max(content.find(word) - SNIPPET_CHARS // 2, 0)
    return content[start : start + SNIPPET_CHARS]
