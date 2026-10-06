"""SQLite 待办表的结构和初始化。

这个模块负责待办数据表，以及最基本的新增、查询、修改和完成操作。
模型工具的 JSON Schema 和 Agent 循环会在后续任务中加入。
"""

import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

TODO_SCHEMA = """
CREATE TABLE IF NOT EXISTS todos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    due_at TEXT,
    priority INTEGER NOT NULL DEFAULT 2 CHECK (priority BETWEEN 1 AND 3),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'completed')),
    project TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_todos_status ON todos(status);
CREATE INDEX IF NOT EXISTS idx_todos_due_at ON todos(due_at);
"""


def initialize_database(db_path: str | Path) -> sqlite3.Connection:
    """创建待办数据库并返回连接。

    ``due_at`` 和时间字段统一保存为 ISO 8601 文本；SQLite 没有独立的日期类型，
    这样可以保留时区信息，也方便后续转换成 Python 的 datetime。
    ``tags`` 暂时保存为 JSON 文本，等工具层实现时再负责转换成字符串列表。
    """
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(TODO_SCHEMA)
    connection.commit()
    return connection


def _now() -> str:
    """返回带时区的当前时间，统一用于 created_at 和 updated_at。"""
    return datetime.now(UTC).isoformat()


def _validate_title(title: str) -> str:
    if not isinstance(title, str) or not title.strip():
        raise ValueError("待办标题不能为空")
    return title.strip()


def _validate_priority(priority: int) -> int:
    if priority not in (1, 2, 3):
        raise ValueError("priority 必须是 1、2 或 3")
    return priority


def _validate_status(status: str) -> str:
    if status not in ("pending", "completed"):
        raise ValueError("status 必须是 pending 或 completed")
    return status


def _encode_tags(tags: Sequence[str] | None) -> str:
    if tags is None:
        return "[]"
    if not all(isinstance(tag, str) and tag.strip() for tag in tags):
        raise ValueError("tags 必须是非空字符串列表")
    return json.dumps([tag.strip() for tag in tags], ensure_ascii=False)


def _decode_row(row: sqlite3.Row) -> dict[str, object]:
    result = dict(row)
    result["tags"] = json.loads(result["tags"])
    return result


def _get_todo(connection: sqlite3.Connection, todo_id: int) -> dict[str, object]:
    row = connection.execute("SELECT * FROM todos WHERE id = ?", (todo_id,)).fetchone()
    if row is None:
        raise KeyError(f"找不到待办：{todo_id}")
    return _decode_row(row)


def add_todo(
    connection: sqlite3.Connection,
    title: str,
    *,
    due_at: str | None = None,
    priority: int = 2,
    status: str = "pending",
    project: str | None = None,
    tags: Sequence[str] | None = None,
) -> dict[str, object]:
    """新增一条待办，并返回保存后的记录。"""
    title = _validate_title(title)
    priority = _validate_priority(priority)
    status = _validate_status(status)
    encoded_tags = _encode_tags(tags)
    now = _now()
    cursor = connection.execute(
        """
        INSERT INTO todos
            (title, due_at, priority, status, project, tags, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (title, due_at, priority, status, project, encoded_tags, now, now),
    )
    connection.commit()
    return _get_todo(connection, cursor.lastrowid)


def list_todos(
    connection: sqlite3.Connection,
    *,
    status: str | None = None,
    priority: int | None = None,
) -> list[dict[str, object]]:
    """查询待办，可按状态或优先级筛选。"""
    conditions: list[str] = []
    parameters: list[object] = []
    if status is not None:
        conditions.append("status = ?")
        parameters.append(_validate_status(status))
    if priority is not None:
        conditions.append("priority = ?")
        parameters.append(_validate_priority(priority))

    query = "SELECT * FROM todos"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY id"
    rows = connection.execute(query, parameters).fetchall()
    return [_decode_row(row) for row in rows]


def update_todo(connection: sqlite3.Connection, todo_id: int, **changes: object) -> dict[str, object]:
    """修改一条待办中调用者明确提供的字段。"""
    allowed_fields = {"title", "due_at", "priority", "status", "project", "tags"}
    unknown_fields = set(changes) - allowed_fields
    if unknown_fields:
        raise ValueError(f"不支持修改这些字段：{', '.join(sorted(unknown_fields))}")
    if not changes:
        raise ValueError("至少要提供一个要修改的字段")

    if "title" in changes:
        changes["title"] = _validate_title(changes["title"])
    if "priority" in changes:
        changes["priority"] = _validate_priority(changes["priority"])
    if "status" in changes:
        changes["status"] = _validate_status(changes["status"])
    if "tags" in changes:
        changes["tags"] = _encode_tags(changes["tags"])

    assignments = [f"{field} = ?" for field in changes]
    parameters = list(changes.values())
    assignments.append("updated_at = ?")
    parameters.extend([_now(), todo_id])
    cursor = connection.execute(
        # assignments 只来自上面的 allowed_fields，不接受调用者传入的字段名。
        f"UPDATE todos SET {', '.join(assignments)} WHERE id = ?",
        parameters,
    )
    if cursor.rowcount == 0:
        connection.rollback()
        raise KeyError(f"找不到待办：{todo_id}")
    connection.commit()
    return _get_todo(connection, todo_id)


def complete_todo(connection: sqlite3.Connection, todo_id: int) -> dict[str, object]:
    """把待办标记为已完成。"""
    return update_todo(connection, todo_id, status="completed")
