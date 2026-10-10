"""Agent 应用层的兼容入口。

新的主程序通过 :mod:`kairos.runtime` 创建并持有会话。本模块暂时保留阶段 1
已经使用的 ``run_prompt``、``run_cli`` 和构造函数名称，避免一次重构破坏现有
调试脚本与外部调用方。
"""

from pathlib import Path
from typing import Any

from .agent_loop import Model
from .config import PROJECT_ROOT, Settings
from .runtime import (
    DEFAULT_SYSTEM_PROMPT,
    AgentRuntime,
    build_model,
    build_tool_handlers,
    create_runtime,
)

__all__ = [
    "AgentRuntime",
    "DEFAULT_SYSTEM_PROMPT",
    "build_model",
    "build_tool_handlers",
    "create_runtime",
    "run_cli",
    "run_prompt",
]


def run_prompt(
    prompt: str,
    *,
    model: Model,
    db_path: str | Path,
    log_path: str | Path | None = None,
) -> tuple[str, list[dict]]:
    """兼容一次性 Agent 调用；正式交互应复用 ``AgentRuntime``。"""

    runtime = create_runtime(model=model, db_path=db_path, log_path=log_path)
    try:
        result = runtime.respond(prompt)
        return result.content, result.messages
    finally:
        runtime.close()


def run_cli(
    prompt: str,
    *,
    settings: Settings | None = None,
    client: Any | None = None,
    db_path: str | Path | None = None,
    log_path: str | Path | None = None,
) -> str:
    """兼容一次性命令行调用；正式主入口由 ``kairos.chat`` 管理会话。"""

    runtime = create_runtime(
        settings=settings,
        client=client,
        db_path=db_path or PROJECT_ROOT / "data" / "kairos.sqlite3",
        log_path=log_path,
    )
    try:
        return runtime.respond(prompt).content
    finally:
        runtime.close()
