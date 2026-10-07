"""Kairos 待办 Agent 的正式编排入口。

这里放可复用的模型、工具和数据库连接逻辑；命令行调试脚本只负责读取输入和打印结果。
"""

from functools import partial
from pathlib import Path
from typing import Any

from .agent_loop import Model, run_tool_loop
from .config import PROJECT_ROOT, Settings
from .model import DeepSeekModel
from .todos import add_todo, complete_todo, initialize_database, list_todos, update_todo
from .tool_schemas import TODO_TOOL_SCHEMAS

SYSTEM_PROMPT = """你是 Kairos，用户的个人助手。回答简洁直接，使用中文。
当用户要求新增、查询、修改或完成待办时，调用对应工具；工具返回结果后再用中文说明结果。
没有必要操作待办时直接回答，不要为了调用工具而调用工具。"""


def build_tool_handlers(connection: Any) -> dict[str, Any]:
    """把数据库连接绑定到四个待办函数，交给 Agent 循环按名称调用。"""
    return {
        "add_todo": partial(add_todo, connection),
        "list_todos": partial(list_todos, connection),
        "update_todo": partial(update_todo, connection),
        "complete_todo": partial(complete_todo, connection),
    }


def run_prompt(prompt: str, *, model: Model, db_path: str | Path) -> tuple[str, list[dict]]:
    """用传入的模型执行一轮待办 Agent 请求。"""
    connection = initialize_database(db_path)
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return run_tool_loop(
            messages,
            model,
            build_tool_handlers(connection),
            TODO_TOOL_SCHEMAS,
        )
    finally:
        connection.close()


def run_cli(
    prompt: str, *, settings: Settings | None = None, client: Any | None = None, db_path: str | Path | None = None
) -> str:
    """初始化真实 DeepSeek 模型并执行一轮请求。"""
    settings = settings or Settings()
    model = DeepSeekModel(settings, client=client)
    database_path = db_path or PROJECT_ROOT / "data" / "kairos.sqlite3"
    answer, _ = run_prompt(prompt, model=model, db_path=database_path)
    return answer
