"""SQLite 待办表的结构和初始化。

这一阶段只负责把数据放在哪里、每个字段是什么定义清楚。
新增、查询、修改和完成待办的函数会在后续任务中实现。
"""

import sqlite3
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
