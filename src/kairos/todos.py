"""SQLite 待办表的结构和初始化。

这个模块负责待办数据表，以及最基本的新增、查询、修改、完成和删除操作。
"""

import json
import re
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
    if type(priority) is not int or priority not in (1, 2, 3):
        raise ValueError("priority 必须是 1、2 或 3")
    return priority


def _validate_status(status: str) -> str:
    if status not in ("pending", "completed"):
        raise ValueError("status 必须是 pending 或 completed")
    return status


def _encode_tags(tags: Sequence[str] | None) -> str:
    if tags is None:
        return "[]"
    if (
        not isinstance(tags, Sequence)
        or isinstance(tags, (str, bytes, bytearray))
        or not all(isinstance(tag, str) and tag.strip() for tag in tags)
    ):
        raise ValueError("tags 必须是非空字符串列表")
    return json.dumps([tag.strip() for tag in tags], ensure_ascii=False)


def _validate_todo_id(todo_id: int) -> int:
    """校验待办 ID，阻止布尔值或字符串被 SQLite 当作整数。

    作用:
        在查询或修改目标记录前限定 ID 类型，避免非法模型参数操作其他待办。

    参数:
        todo_id: 必须是大于零的整数，不能是 bool。

    返回:
        原始合法 ID（int）。

    异常:
        ValueError: ID 不是严格的正整数。

    行为说明:
        1. 在执行 SQL 前检查，不做隐式类型转换。
    """
    if type(todo_id) is not int or todo_id < 1:
        raise ValueError("todo_id 必须是正整数，不能是布尔值或字符串")
    return todo_id


def _validate_due_at(due_at: str | None) -> str | None:
    """校验并规范化带明确时区的截止时间。

    作用:
        在新增或修改待办前拒绝相对日期、无时区时间和无效日期，不猜测用户时区。

    参数:
        due_at: None 表示没有截止时间；字符串须包含日期、时分秒及 Z 或 ±HH:MM 时区。

    返回:
        规范化 ISO 8601 字符串或 None，保留输入所指定的时区偏移。

    异常:
        ValueError: 输入不符合规定格式，或日期、时间字段无效。

    行为说明:
        1. 先检查类型和完整格式，再用 datetime 校验真实日期。
        2. 所有校验均在 SQL 写入之前完成；不为相对时间补充当前日期。
    """
    if due_at is None:
        return None
    pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)"
    if not isinstance(due_at, str) or re.fullmatch(pattern, due_at, re.IGNORECASE) is None:
        raise ValueError("due_at 必须是带时区的 ISO 8601 时间，例如 2026-10-18T15:00:00+08:00，或 null")
    try:
        return datetime.fromisoformat(due_at.replace("z", "Z")).isoformat()
    except ValueError:
        raise ValueError("due_at 的日期或时间无效") from None


def _validate_project(project: str | None) -> str | None:
    """校验项目字段，防止 SQLite 隐式转换非法类型。

    作用:
        在待办新增和修改的写入边界检查所属项目。

    参数:
        project: 项目字符串或 None；None 表示不关联项目。

    返回:
        原始项目值（str | None）。

    异常:
        ValueError: 项目既不是字符串也不是 None。

    行为说明:
        1. 仅校验类型，不创建项目，也不转换名称。
    """
    if project is not None and not isinstance(project, str):
        raise ValueError("project 必须是字符串或 null")
    return project


def _decode_row(row: sqlite3.Row) -> dict[str, object]:
    result = dict(row)
    result["tags"] = json.loads(result["tags"])
    return result


def _get_todo(connection: sqlite3.Connection, todo_id: int) -> dict[str, object]:
    todo_id = _validate_todo_id(todo_id)
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
    due_at = _validate_due_at(due_at)
    project = _validate_project(project)
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
    todo_id = _validate_todo_id(todo_id)
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
    if "due_at" in changes:
        changes["due_at"] = _validate_due_at(changes["due_at"])
    if "project" in changes:
        changes["project"] = _validate_project(changes["project"])

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


def delete_todo(connection: sqlite3.Connection, todo_id: int) -> dict[str, object]:
    """按 ID 永久删除一条待办，并返回被删除的记录。"""
    todo = _get_todo(connection, todo_id)
    connection.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
    connection.commit()
    return todo
