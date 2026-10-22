"""历史会话工具：让模型搜索和阅读过去的对话。只读，不切换当前会话。"""

import sqlite3

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

RULES = "用户问起以前讨论过的事情时，先用 search_history 找到相关会话，再用 read_session 阅读上下文。"

PAGE_SIZE = 20
SNIPPET_CHARS = 200


def history_tools(db: sqlite3.Connection) -> Toolset:
    def search_history(query: str, limit: int = 5) -> list[dict]:
        words = query.split()
        # 每个关键词一个 LIKE 条件，全部满足才算匹配。content 从消息 JSON 里取。
        conditions = " AND ".join("json_extract(m.message, '$.content') LIKE ?" for _ in words)
        rows = db.execute(
            f"""SELECT m.session_id, s.title, m.role, m.created_at,
                       json_extract(m.message, '$.content') AS content
                FROM messages m JOIN sessions s ON s.id = m.session_id
                WHERE m.role IN ('user', 'assistant') AND {conditions}
                ORDER BY m.id DESC LIMIT ?""",
            [*(f"%{word}%" for word in words), limit],
        ).fetchall()
        return [
            {
                "session_id": row["session_id"],
                "title": row["title"],
                "role": row["role"],
                "time": row["created_at"],
                "snippet": _snippet(row["content"], words[0]),
            }
            for row in rows
        ]

    def read_session(session_id: str, offset: int = 0) -> dict:
        rows = db.execute(
            """SELECT role, created_at, json_extract(message, '$.content') AS content
               FROM messages
               WHERE session_id = ? AND role IN ('user', 'assistant') AND content != ''
               ORDER BY id LIMIT ? OFFSET ?""",
            (session_id, PAGE_SIZE + 1, offset),
        ).fetchall()
        messages = [{"role": row["role"], "time": row["created_at"], "content": row["content"]} for row in rows]
        has_more = len(messages) > PAGE_SIZE
        return {
            "messages": messages[:PAGE_SIZE],
            "next_offset": offset + PAGE_SIZE if has_more else None,
        }

    return Toolset(
        schemas=[SEARCH_HISTORY, READ_SESSION],
        handlers={"search_history": search_history, "read_session": read_session},
        instructions=[RULES],
    )


def _snippet(content: str, word: str) -> str:
    """截取关键词附近的一段文字。"""
    start = max(content.find(word) - SNIPPET_CHARS // 2, 0)
    return content[start : start + SNIPPET_CHARS]
