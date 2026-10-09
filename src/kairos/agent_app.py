"""Kairos 待办 Agent 的正式编排入口。

这里放可复用的模型、工具和数据库连接逻辑；命令行调试脚本只负责读取输入和打印结果。
"""

from functools import partial
from pathlib import Path
from typing import Any

from .agent_loop import Model, run_single_tool_turn
from .config import PROJECT_ROOT, Settings
from .model import AnthropicMessagesModel, OpenAICompatibleModel
from .observability import JsonlLogger
from .todos import add_todo, complete_todo, initialize_database, list_todos, update_todo
from .tool_schemas import TODO_TOOL_SCHEMAS

SYSTEM_PROMPT = """你是 Kairos，用户的个人助手。回答简洁直接，使用中文。
当用户要求新增、查询、修改或完成待办时，调用对应工具；工具返回结果后再用中文说明结果。
没有必要操作待办时直接回答，不要为了调用工具而调用工具。"""


def _anthropic_base_url(settings: Settings) -> str | None:
    """返回 Anthropic SDK 使用的根地址，并兼容现有的 OpenAI 网关配置。"""
    explicit = getattr(settings, "anthropic_base_url", None)
    if explicit:
        return explicit.rstrip("/")

    shared = getattr(settings, "openai_base_url", None) or getattr(settings, "deepseek_base_url", None)
    if not shared:
        return None
    shared = shared.rstrip("/")
    if shared.endswith("/v1"):
        return shared[:-3]
    return shared


def build_tool_handlers(connection: Any) -> dict[str, Any]:
    """把数据库连接绑定到四个待办函数，交给 Agent 循环按名称调用。"""
    return {
        "add_todo": partial(add_todo, connection),
        "list_todos": partial(list_todos, connection),
        "update_todo": partial(update_todo, connection),
        "complete_todo": partial(complete_todo, connection),
    }


def build_model(settings: Settings, client: Any | None = None) -> Model:
    """根据配置选择协议适配器；工具循环不需要知道具体供应商。"""
    provider = getattr(settings, "kairos_provider", "openai")
    if provider == "openai":
        extra_body = {"thinking": {"type": "disabled"}}
        if not getattr(settings, "kairos_model", "").startswith("deepseek"):
            extra_body = None
        return OpenAICompatibleModel(
            model=settings.kairos_model,
            api_key=getattr(settings, "openai_api_key", None) or getattr(settings, "deepseek_api_key", None),
            base_url=getattr(settings, "openai_base_url", None) or getattr(settings, "deepseek_base_url", None),
            client=client,
            extra_body=extra_body,
        )
    if provider == "anthropic":
        return AnthropicMessagesModel(
            settings.kairos_model,
            api_key=getattr(settings, "anthropic_api_key", None)
            or getattr(settings, "openai_api_key", None)
            or getattr(settings, "deepseek_api_key", None),
            base_url=_anthropic_base_url(settings),
            client=client,
            max_tokens=getattr(settings, "anthropic_max_tokens", 4096),
        )
    raise ValueError(f"不支持的模型协议：{provider}")


def run_prompt(
    prompt: str,
    *,
    model: Model,
    db_path: str | Path,
    log_path: str | Path | None = None,
) -> tuple[str, list[dict]]:
    """用传入的模型执行一轮待办 Agent 请求。"""
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
            logger=JsonlLogger(log_path) if log_path else None,
        )
    finally:
        connection.close()


def run_cli(
    prompt: str,
    *,
    settings: Settings | None = None,
    client: Any | None = None,
    db_path: str | Path | None = None,
    log_path: str | Path | None = None,
) -> str:
    """根据配置初始化模型适配器并执行一轮请求。"""
    settings = settings or Settings()
    model = build_model(settings, client=client)
    database_path = db_path or PROJECT_ROOT / "data" / "kairos.sqlite3"
    answer, _ = run_prompt(prompt, model=model, db_path=database_path, log_path=log_path)
    return answer
