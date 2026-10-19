"""将独立待办能力封装为普通工具定义与可调用函数。"""

import sqlite3
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import Any

from .storage import add_todo, complete_todo, delete_todo, initialize_schema, list_todos, update_todo
from .tool_schemas import TODO_TOOL_SCHEMAS

TODO_INSTRUCTIONS = (
    "当用户要求新增、查询、修改、完成或删除待办时，调用对应工具；工具返回结果后再用中文说明结果。",
    "删除会永久移除记录，只在用户明确要求删除时执行；未提供 ID 时先查询定位，目标不明确时先询问，不猜测 ID。",
)


def _database_tool(connection: sqlite3.Connection, handler: Callable[..., object], **arguments: Any) -> object:
    """为数据库工具调用提供提交或回滚边界。

    作用:
        作为数据库工具的包装器，接收调用方参数并执行工具。
        工具异常时回滚尚未提交的事务，便于宿主随后独立保存错误结果。

    参数:
        connection: 支持上下文管理的数据库连接，作为第一个参数传给工具。
        handler: 实际执行数据库操作的工具函数。
        arguments: 工具的关键字参数，原样转交 handler。

    返回:
        handler 的原始返回值；本函数不转换工具结果。

    行为说明:
        1. 进入连接的上下文，在其中调用 handler(connection, **arguments)。
        2. 正常退出时提交尚未提交的事务；异常退出时回滚尚未提交的事务。
        3. 此边界不撤销工具自行提前提交的写入，也不处理数据库以外的副作用。
    """
    with connection:
        return handler(connection, **arguments)


def build_tools(connection: sqlite3.Connection) -> dict[str, Any]:
    """初始化待办存储并将连接绑定到五个待办工具。

    作用:
        作为宿主加载待办能力的入口，返回普通 Python 数据，不依赖宿主的注册表类型。

    参数:
        connection: 无活跃事务且 row_factory 为 sqlite3.Row 的 SQLite 连接，由宿主管理生命周期。

    返回:
        含 handlers、schemas、instructions 的字典；分别是执行函数映射、工具定义列表和提示规则元组。

    行为说明:
        1. 通过 initialize_schema 幂等建立自身待办结构，连接契约由它校验。
        2. 工具函数绑定到同一连接，并保留失败时回滚未提交事务的边界。
        3. 每次返回独立的 Schema 副本，避免调用方修改影响后续加载。
        4. 不关闭连接，不调用模型，不启动子进程或 MCP 服务。
    """
    initialize_schema(connection)
    handlers = {
        "add_todo": partial(_database_tool, connection, add_todo),
        "list_todos": partial(_database_tool, connection, list_todos),
        "update_todo": partial(_database_tool, connection, update_todo),
        "complete_todo": partial(_database_tool, connection, complete_todo),
        "delete_todo": partial(_database_tool, connection, delete_todo),
    }
    return {
        "handlers": handlers,
        "schemas": deepcopy(TODO_TOOL_SCHEMAS),
        "instructions": TODO_INSTRUCTIONS,
    }
