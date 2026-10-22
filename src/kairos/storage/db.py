"""打开 Kairos 的 SQLite 数据库。各模块自己建自己的表。"""

import sqlite3
from pathlib import Path


def open_db(path: str | Path) -> sqlite3.Connection:
    """打开（不存在就创建）数据库。行可以用列名访问，例如 row["title"]。

    check_same_thread=False：终端和微信两个线程共用这个连接，由 Services.lock 保证不会同时使用。
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection
