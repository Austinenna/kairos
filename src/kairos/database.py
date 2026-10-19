"""创建 Kairos 的 SQLite 连接，不初始化任何业务模块的表。"""

import sqlite3
from pathlib import Path


def open_database(db_path: str | Path) -> sqlite3.Connection:
    """打开应用数据库，保留已有数据和文件权限。

    作用:
        为运行时提供通用连接；会话存储和可选工具模块分别初始化自己的表。

    参数:
        db_path: 调用方指定的数据库路径，支持用户目录缩写。

    返回:
        使用 sqlite3.Row 的连接 (sqlite3.Connection)，由调用方负责关闭。

    异常:
        捕获的原异常: 打开连接后设置失败时，关闭连接并重新抛出。

    行为说明:
        1. 创建缺失的目录，并为新目录和数据库文件设置仅当前用户可访问的权限。
        2. 已有文件的内容和权限不变，不创建业务表、不启动事务。
        3. 打开后设置失败时关闭连接，再传播原异常。
    """
    path = Path(db_path).expanduser()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    existed = path.exists()
    connection = sqlite3.connect(path)
    try:
        if not existed:
            path.chmod(0o600)
        connection.row_factory = sqlite3.Row
        return connection
    except BaseException:
        connection.close()
        raise
