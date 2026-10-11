"""Kairos 的统一运行时。

运行时把模型、Agent 循环、工具、数据库和当前对话状态组装成一个可复用的
交互会话。CLI 只负责输入和展示；Agent 循环只负责执行模型与工具之间的流程。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from .agent_loop import EventLogger, Model, ToolHandler, run_agent_turn
from .config import PROJECT_ROOT, Settings
from .observability import JsonlLogger
from .pricing import Usage
from .providers import AnthropicMessagesModel, OpenAICompatibleModel
from .todos import add_todo, complete_todo, initialize_database, list_todos, update_todo
from .tool_schemas import TODO_TOOL_SCHEMAS

DEFAULT_SYSTEM_PROMPT = """你是 Kairos，用户的个人助手。回答简洁直接，使用中文。
当用户要求新增、查询、修改或完成待办时，调用对应工具；工具返回结果后再用中文说明结果。
没有必要操作待办时直接回答，不要为了调用工具而调用工具。"""


@dataclass(frozen=True)
class ToolRegistry:
    """模型可见的工具定义和实际执行函数。"""

    handlers: dict[str, ToolHandler]
    schemas: list[dict]


@dataclass(frozen=True)
class AgentTurnResult:
    """一次用户输入完成后的结果。"""

    content: str
    messages: list[dict]
    usage: Usage | None
    finish_reason: str | None
    events: list[dict[str, Any]]


def build_tool_handlers(connection: Any) -> dict[str, ToolHandler]:
    """把数据库连接绑定到待办工具函数。"""

    return {
        "add_todo": partial(add_todo, connection),
        "list_todos": partial(list_todos, connection),
        "update_todo": partial(update_todo, connection),
        "complete_todo": partial(complete_todo, connection),
    }


def build_tool_registry(connection: Any) -> ToolRegistry:
    """创建当前应用使用的待办工具注册表。"""

    return ToolRegistry(handlers=build_tool_handlers(connection), schemas=list(TODO_TOOL_SCHEMAS))


def _anthropic_base_url(settings: Settings) -> str | None:
    """返回 Anthropic SDK 使用的根地址。"""

    explicit = getattr(settings, "anthropic_base_url", None)
    if explicit:
        return explicit.rstrip("/")

    shared = settings.gateway_base_url
    if not shared:
        return None
    shared = shared.rstrip("/")
    if shared.endswith("/v1"):
        return shared[:-3]
    return shared


def build_model(
    settings: Settings,
    client: Any | None = None,
    *,
    thinking_enabled: bool = False,
) -> Model:
    """根据配置创建模型适配器，供 Agent Loop 使用。"""

    provider = settings.kairos_provider
    if provider == "openai":
        extra_body = None
        if settings.kairos_model.startswith("deepseek"):
            extra_body = {"thinking": {"type": "enabled" if thinking_enabled else "disabled"}}
        return OpenAICompatibleModel(
            model=settings.kairos_model,
            api_key=settings.kairos_api_key,
            base_url=settings.gateway_base_url,
            client=client,
            extra_body=extra_body,
        )
    if provider == "anthropic":
        return AnthropicMessagesModel(
            settings.kairos_model,
            api_key=settings.kairos_api_key,
            base_url=_anthropic_base_url(settings),
            client=client,
            max_tokens=settings.anthropic_max_tokens,
        )
    raise ValueError(f"不支持的模型协议：{provider}")


def _aggregate_usage(events: list[dict[str, Any]]) -> Usage | None:
    """汇总一次 Agent 回合中所有模型请求的用量。"""

    response_events = [event for event in events if event.get("event") == "model_response"]
    usages = [event.get("usage") for event in response_events if event.get("usage")]
    if not usages:
        return None

    return Usage(
        input_tokens=sum(int(usage.get("prompt_tokens") or 0) for usage in usages),
        cached_input_tokens=sum(int(usage.get("cached_input_tokens") or 0) for usage in usages),
        output_tokens=sum(int(usage.get("completion_tokens") or 0) for usage in usages),
        reasoning_tokens=sum(int(usage.get("reasoning_tokens") or 0) for usage in usages),
    )


def _finish_reason(events: list[dict[str, Any]]) -> str | None:
    response_events = [event for event in events if event.get("event") == "model_response"]
    if not response_events:
        return None
    return (response_events[-1].get("response") or {}).get("finish_reason")


class AgentRuntime:
    """一个交互式 Agent 会话的中央协调器。"""

    def __init__(
        self,
        *,
        model: Model,
        connection: Any,
        tool_registry: ToolRegistry,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        logger: EventLogger | None = None,
        max_tool_rounds: int = 8,
        thinking_enabled: bool = False,
    ) -> None:
        self.model = model
        self.connection = connection
        self.tool_registry = tool_registry
        self.logger = logger
        self.max_tool_rounds = max_tool_rounds
        self.thinking_enabled = thinking_enabled
        self.system_prompt = system_prompt
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self.last_result: AgentTurnResult | None = None
        self._closed = False

    @property
    def model_name(self) -> str:
        return str(getattr(self.model, "model", "unknown"))

    def respond(self, prompt: str) -> AgentTurnResult:
        """处理一条用户消息，并保留本次回合产生的完整消息历史。"""

        if self._closed:
            raise RuntimeError("AgentRuntime 已经关闭")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("用户消息不能为空")

        original_length = len(self.messages)
        self.messages.append({"role": "user", "content": prompt})
        events: list[dict[str, Any]] = []

        def record(event: dict[str, Any]) -> None:
            events.append(event)
            if self.logger:
                self.logger(event)

        try:
            content, messages = run_agent_turn(
                self.messages,
                self.model,
                self.tool_registry.handlers,
                self.tool_registry.schemas,
                max_tool_rounds=self.max_tool_rounds,
                logger=record,
            )
        except Exception:
            del self.messages[original_length:]
            raise

        result = AgentTurnResult(
            content=content,
            messages=list(messages),
            usage=_aggregate_usage(events),
            finish_reason=_finish_reason(events),
            events=events,
        )
        self.last_result = result
        return result

    def reset(self, system_prompt: str | None = None) -> None:
        """清空当前对话，但保留模型、工具和数据库连接。"""

        if self._closed:
            raise RuntimeError("AgentRuntime 已经关闭")
        if system_prompt is not None:
            self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.last_result = None

    def set_thinking(self, enabled: bool) -> None:
        """更新支持该参数的模型的思考模式。"""

        self.thinking_enabled = enabled
        if isinstance(self.model, OpenAICompatibleModel) and self.model.model.startswith("deepseek"):
            self.model.extra_body = {"thinking": {"type": "enabled" if enabled else "disabled"}}

    def history(self) -> list[dict]:
        """返回当前对话历史的浅拷贝，供 CLI 查看。"""

        return list(self.messages)

    def close(self) -> None:
        """关闭运行时持有的数据库连接。"""

        if not self._closed:
            self.connection.close()
            self._closed = True

    def __enter__(self) -> AgentRuntime:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def create_runtime(
    *,
    settings: Settings | None = None,
    client: Any | None = None,
    model: Model | None = None,
    db_path: str | Path | None = None,
    log_path: str | Path | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    thinking_enabled: bool = False,
) -> AgentRuntime:
    """创建一个完整的 AgentRuntime，并负责初始化其依赖。"""

    if model is None:
        settings = settings or Settings()
    database_path = db_path or PROJECT_ROOT / "data" / "kairos.sqlite3"
    connection = initialize_database(database_path)
    try:
        runtime_model = model or build_model(settings, client=client, thinking_enabled=thinking_enabled)
        logger = JsonlLogger(log_path) if log_path else None
        return AgentRuntime(
            model=runtime_model,
            connection=connection,
            tool_registry=build_tool_registry(connection),
            system_prompt=system_prompt,
            logger=logger,
            thinking_enabled=thinking_enabled,
        )
    except Exception:
        connection.close()
        raise
