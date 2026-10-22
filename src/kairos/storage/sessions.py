"""会话存储：两张表，sessions 记会话信息，messages 按顺序记完整消息和用量。"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from ..images import text_of
from ..models import Usage

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    provider   TEXT NOT NULL,              -- openai / anthropic
    model      TEXT NOT NULL,
    thinking   INTEGER NOT NULL DEFAULT 0, -- 思考模式开关
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,  -- 也是消息顺序
    session_id          TEXT NOT NULL REFERENCES sessions(id),
    role                TEXT NOT NULL,
    message             TEXT NOT NULL,     -- 完整消息 JSON，恢复时原样发回模型
    -- 以下只有助手消息才有：产生这条消息的那次模型请求的用量和费用
    input_tokens        INTEGER,
    cached_input_tokens INTEGER,
    output_tokens       INTEGER,
    reasoning_tokens    INTEGER,
    cost_usd            REAL,
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""

DEFAULT_TITLE = "新会话"


@dataclass
class Session:
    id: str
    title: str
    provider: str
    model: str
    thinking: bool
    created_at: str
    updated_at: str


@dataclass
class SessionTotals:
    """一个会话累计的用量和费用。"""

    usage: Usage
    cost_usd: float
    requests: int  # 模型请求次数
    last_input_tokens: int | None  # 最近一次请求的完整输入量，用来显示上下文占用


class SessionStore:
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.db.executescript(SCHEMA)

    def create(self, *, provider: str, model: str, system_prompt: str, thinking: bool = False) -> Session:
        """新建会话，并把 system 提示词存为第一条消息。"""
        now = _now()
        session_id = uuid4().hex
        with self.db:
            self.db.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, DEFAULT_TITLE, provider, model, thinking, now, now),
            )
        self.add_message(session_id, {"role": "system", "content": system_prompt})
        return self.get(session_id)

    def get(self, reference: str) -> Session:
        """按完整 ID 或唯一前缀查找会话。找不到或前缀不唯一时抛出 ValueError。"""
        rows = self.db.execute("SELECT * FROM sessions WHERE id LIKE ? LIMIT 2", (reference + "%",)).fetchall()
        if len(rows) != 1:
            raise ValueError(f"找不到唯一匹配 {reference} 的会话")
        return _session(rows[0])

    def recent(self, limit: int = 20) -> list[Session]:
        """最近活动的会话在前；没发过消息的空会话不列出。"""
        rows = self.db.execute(
            """SELECT * FROM sessions s
               WHERE EXISTS (SELECT 1 FROM messages m WHERE m.session_id = s.id AND m.role = 'user')
               ORDER BY updated_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [_session(row) for row in rows]

    def latest(self) -> Session | None:
        sessions = self.recent(limit=1)
        return sessions[0] if sessions else None

    def rename(self, session_id: str, title: str) -> None:
        with self.db:
            self.db.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))

    def set_thinking(self, session_id: str, enabled: bool) -> None:
        with self.db:
            self.db.execute("UPDATE sessions SET thinking = ? WHERE id = ?", (enabled, session_id))

    def add_message(
        self, session_id: str, message: dict, usage: Usage | None = None, cost_usd: float | None = None
    ) -> None:
        """追加一条消息。会话的第一条用户消息同时用作默认标题。"""
        usage = usage or Usage()
        now = _now()
        with self.db:
            self.db.execute(
                """INSERT INTO messages (session_id, role, message, input_tokens, cached_input_tokens,
                                         output_tokens, reasoning_tokens, cost_usd, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    message["role"],
                    json.dumps(message, ensure_ascii=False),
                    usage.input_tokens,
                    usage.cached_input_tokens,
                    usage.output_tokens,
                    usage.reasoning_tokens,
                    cost_usd,
                    now,
                ),
            )
            self.db.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
            if message["role"] == "user":
                self.db.execute(
                    "UPDATE sessions SET title = ? WHERE id = ? AND title = ?",
                    (text_of(message["content"]).strip()[:30], session_id, DEFAULT_TITLE),
                )

    def load_messages(self, session_id: str) -> list[dict]:
        rows = self.db.execute("SELECT message FROM messages WHERE session_id = ? ORDER BY id", (session_id,))
        return [json.loads(row["message"]) for row in rows]

    def totals(self, session_id: str) -> SessionTotals:
        """汇总会话的用量和费用。SUM 会跳过空值，所以缺项的请求不计入。"""
        row = self.db.execute(
            """SELECT SUM(input_tokens) AS input, SUM(cached_input_tokens) AS cached,
                      SUM(output_tokens) AS output, SUM(reasoning_tokens) AS reasoning,
                      SUM(cost_usd) AS cost, COUNT(*) AS requests
               FROM messages WHERE session_id = ? AND role = 'assistant'""",
            (session_id,),
        ).fetchone()
        last = self.db.execute(
            """SELECT input_tokens FROM messages
               WHERE session_id = ? AND input_tokens IS NOT NULL ORDER BY id DESC LIMIT 1""",
            (session_id,),
        ).fetchone()
        return SessionTotals(
            usage=Usage(row["input"] or 0, row["cached"] or 0, row["output"] or 0, row["reasoning"] or 0),
            cost_usd=row["cost"] or 0.0,
            requests=row["requests"],
            last_input_tokens=last["input_tokens"] if last else None,
        )


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _session(row: sqlite3.Row) -> Session:
    return Session(**{**dict(row), "thinking": bool(row["thinking"])})
