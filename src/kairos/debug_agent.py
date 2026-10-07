"""阶段 1 的真实模型调试入口。

示例：

    uv run python -m kairos.debug_agent "把学习 Agent 加入待办"

该入口只执行一轮请求，数据库保存在 ``data/kairos.sqlite3``，便于观察模型、
工具 Schema、SQLite 和 agent 循环之间的完整连接。
"""

import argparse
from functools import partial
from pathlib import Path
from typing import Any

from .agent_loop import Model, run_single_tool_turn
from .config import PROJECT_ROOT, Settings
from .model import DeepSeekModel
from .todos import add_todo, complete_todo, initialize_database, list_todos, update_todo
from .tool_schemas import TODO_TOOL_SCHEMAS

SYSTEM_PROMPT = """你是 Kairos，用户的个人助手。回答简洁直接，使用中文。
当用户要求新增、查询、修改或完成待办时，调用对应工具；工具返回结果后再用中文说明结果。
没有必要操作待办时直接回答，不要为了调用工具而调用工具。"""


def build_tool_handlers(connection: Any) -> dict[str, Any]:
    """把数据库连接绑定到四个待办函数，交给 agent 循环按名称调用。"""
    return {
        "add_todo": partial(add_todo, connection),
        "list_todos": partial(list_todos, connection),
        "update_todo": partial(update_todo, connection),
        "complete_todo": partial(complete_todo, connection),
    }


def run_prompt(prompt: str, *, model: Model, db_path: str | Path) -> tuple[str, list[dict]]:
    """用传入的模型执行一轮待办 Agent 请求。

    ``model`` 可替换成 Fake Model，因此这层可以在没有网络和密钥时测试完整工具闭环。
    """
    connection = initialize_database(db_path)
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return run_single_tool_turn(
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="用真实 DeepSeek 模型调试 Kairos 待办工具循环")
    parser.add_argument("prompt", nargs="*", help="本轮要发送给 Kairos 的问题；省略时从 stdin 读取")
    args = parser.parse_args(argv)
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        prompt = input("你 › ").strip()
    if not prompt:
        parser.error("请输入要发送的问题")

    print(run_cli(prompt))


if __name__ == "__main__":
    main()
