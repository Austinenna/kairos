"""Kairos 的统一运行时。

运行时把模型、Agent 循环、工具、数据库和当前对话状态组装成一个可复用的
交互会话。CLI 只负责输入和展示；Agent 循环只负责执行模型与工具之间的流程。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from . import config
from .agent_loop import AgentLoopLimitError, EventLogger, Model, run_agent_turn, sanitize_error
from .cancellation import CancellationToken, cancellation_scope, raise_if_cancelled
from .config import Settings, ToolSettings
from .context import ContextBuilder
from .database import open_database
from .execution import TurnRecorder
from .history import HistoryReader
from .history_tools import HISTORY_TOOL_SCHEMAS
from .images import ImageInput, redact_images, validate_image_budget
from .memory import MemoryStore
from .memory_extraction import ExtractionJob, MemoryExtractor
from .memory_tools import PRIMARY_MEMORY_TOOL_SCHEMAS
from .observability import JsonlLogger
from .pricing import Usage, usage_from_normalized
from .providers import AnthropicMessagesModel, OpenAICompatibleModel
from .sessions import SessionInfo, SessionStore
from .tools import ToolRegistry, load_tool_registry, prepare_tool_messages
from .usage_stats import SessionUsage, summarize_usage

DEFAULT_SYSTEM_PROMPT = """你是 Kairos，用户的个人助手。回答简洁直接，使用中文。
需要执行操作时，根据当前可用能力调用对应工具；工具返回结果后再用中文说明结果。
不需要工具时直接回答，不要为了调用工具而调用工具。"""


@dataclass(frozen=True)
class ToolResult:
    """一次工具调用的结构化结果，供上层展示或记录。"""

    call_id: str
    name: str
    value: object
    content: str
    step_id: str | None = None


@dataclass(frozen=True)
class AgentTurnResult:
    """一次用户输入完成后的结果。"""

    content: str
    messages: list[dict]
    usage: Usage | None
    finish_reason: str | None
    events: list[dict[str, Any]]
    tool_results: list[ToolResult]
    session_id: str | None = None
    turn_id: str | None = None
    step_ids: tuple[str, ...] = ()
    last_request_usage: Usage | None = None
    session_usage: SessionUsage | None = None


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


def _build_extraction_model(settings: Settings, *, thinking_enabled: bool = False) -> Model:
    """创建后台独立模型，并为 DeepSeek 记忆提取启用 JSON 输出。

    作用:
        在后台模型构建边界设置输出格式，保持前台普通对话的请求参数不变。

    参数:
        settings: 当前回合的模型与认证配置快照。
        thinking_enabled: 当前回合的思考开关，原样交给模型工厂。

    返回:
        独立的模型适配器（Model）；DeepSeek Chat Completions 启用 json_object 模式。

    行为说明:
        1. 沿用现有工厂创建模型，只对已确认支持 JSON 模式的 DeepSeek 协议组合加参。
        2. 合并额外参数并保留 thinking；不修改前台模型、网关配置或自定义注入工厂。
    """
    model = build_model(settings, thinking_enabled=thinking_enabled)
    if isinstance(model, OpenAICompatibleModel) and model.model.startswith("deepseek"):
        model.extra_body = {**(model.extra_body or {}), "response_format": {"type": "json_object"}}
    return model


def _usage_from_event(event: dict[str, Any]) -> Usage | None:
    """提取一次模型响应的用量，保留未返回的字段。

    作用:
        把观察事件转换为展示和费用计算使用的数据，避免缺失量或非法计数变成零。

    参数:
        event: model_response 事件；usage 使用适配器归一化后的字段名。

    返回:
        Usage 或 None；每个计数仅接受非负整数，未知字段保留 None。

    行为说明:
        1. 没有用量对象时返回 None；否则独立读取四个字段。
        2. 不推算缓存或思考量，不修改事件，不因统计失败打断已完成回合。
    """
    return usage_from_normalized(event.get("usage"))


def _aggregate_usage(events: list[dict[str, Any]]) -> Usage | None:
    """汇总本轮所有请求的用量，缺项时保留未知。

    作用:
        生成一次用户回合的累计消耗，用于展示与估算费用，不表示上下文长度。

    参数:
        events: 按执行顺序排列的观察事件列表。

    返回:
        累计 Usage 或 None；某字段在任一次请求中未知，则该字段的累计值也未知。

    行为说明:
        1. 保留所有模型响应，包括未报告用量的响应。
        2. 逐字段求和；全部请求都没有用量时返回 None，不把部分已知值冒充总量。
    """

    response_events = [event for event in events if event.get("event") == "model_response"]
    usages = [_usage_from_event(event) for event in response_events]
    if not any(usage is not None for usage in usages):
        return None

    totals = {}
    for name in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"):
        counts = [getattr(usage, name, None) for usage in usages]
        totals[name] = sum(counts) if all(count is not None for count in counts) else None
    return Usage(**totals)


def _last_request_usage(events: list[dict[str, Any]]) -> Usage | None:
    """读取最后一次模型请求的用量，不借用前一步数据。

    作用:
        为最近请求的完整输入占比提供依据，避免把累计消耗当作上下文长度。

    参数:
        events: 按执行顺序排列的观察事件列表。

    返回:
        最后一个 model_response 的 Usage，未报告时返回 None。

    行为说明:
        1. 倒序找到最后一个响应后立即转换；即使该响应没有用量也不继续回退。
    """
    for event in reversed(events):
        if event.get("event") == "model_response":
            return _usage_from_event(event)
    return None


def _finish_reason(events: list[dict[str, Any]]) -> str | None:
    response_events = [event for event in events if event.get("event") == "model_response"]
    if not response_events:
        return None
    return (response_events[-1].get("response") or {}).get("finish_reason")


def _tool_results(events: list[dict[str, Any]]) -> list[ToolResult]:
    results: list[ToolResult] = []
    for event in events:
        if event.get("event") != "tool_result":
            continue
        content = str(event.get("result", ""))
        try:
            value: object = json.loads(content)
        except json.JSONDecodeError:
            value = content
        results.append(
            ToolResult(
                call_id=str(event.get("tool_call_id", "")),
                name=str(event.get("tool_name", "")),
                value=value,
                content=content,
                step_id=event.get("step_id"),
            )
        )
    return results


def _model_provider(model: Model) -> str:
    """识别模型适配器对应的会话协议名称。

    作用:
        为新建会话的协议记录和恢复时的兼容性检查提供统一名称，不发起模型请求。

    参数:
        model: 当前使用或调用方注入的模型对象。

    返回:
        OpenAICompatibleModel 返回 "openai"，AnthropicMessagesModel 返回
        "anthropic"，其他对象返回 "custom"。

    行为说明:
        1. 按适配器类型依次判断，子类也使用对应协议名称。
        2. 未识别的模型统一标记为 custom，不尝试从名称推断协议。
    """

    if isinstance(model, OpenAICompatibleModel):
        return "openai"
    if isinstance(model, AnthropicMessagesModel):
        return "anthropic"
    return "custom"


def _supports_thinking(model: Model) -> bool:
    """判断当前适配器是否已接入思考开关。

    作用:
        为新建、恢复和手动切换提供统一判断，不推断模型本身的推理能力。

    参数:
        model: 待使用的模型适配器；恢复时必须传入目标模型。

    返回:
        是否为已实现 thinking 请求参数的 DeepSeek OpenAI 兼容适配器（bool）。

    行为说明:
        1. 同时检查适配器类型和模型名称；自定义模型不会仅凭名称被视为已接入。
    """

    return isinstance(model, OpenAICompatibleModel) and model.model.startswith("deepseek")


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
        session_store: SessionStore | None = None,
        session_info: SessionInfo | None = None,
        model_settings: Settings | None = None,
        model_client: Any | None = None,
        rebuild_model: bool = False,
        owns_model_client: bool = False,
        memory_store: MemoryStore | None = None,
        auto_memory: bool = True,
        extraction_model_factory: Callable[[], Model] | None = None,
    ) -> None:
        """组装运行依赖并激活新建或已有会话。

        作用:
            接收工厂或调用方提供的模型、数据库和工具，建立可执行回合的运行时。
            新建时保存会话及系统消息；恢复时加载历史，并按需修补中断记录。

        参数:
            model: 已创建的模型适配器；直接传入本构造函数时，由调用方保证与
                session_info 的协议和模型兼容，构造函数不执行该兼容性检查。
            connection: 供运行时和数据库工具使用的连接，close 时由运行时关闭。
            tool_registry: 模型可见的工具定义及实际执行函数。
            system_prompt: 新会话的系统提示词；恢复时改用历史中的系统提示词。
            logger: 可选的事件记录回调，执行回合时接收观察事件。
            max_tool_rounds: 包含工具调用的模型响应轮数上限，构造时不校验取值。
                同一响应中的多个工具只计一轮，最终纯文字回答不计入此上限。
            thinking_enabled: 新会话的思考开关；恢复时采用会话保存的值。
            session_store: 可选的会话存储；省略时使用 connection 初始化。
                注入时由调用方保证它与运行时使用兼容的连接。
            session_info: 要恢复的会话元信息；省略时立即创建新会话。
            model_settings: 重建模型时使用的当前配置，包括认证信息。
            model_client: 工厂创建适配器时使用的外部客户端，供切换模型时复用。
            rebuild_model: 是否允许恢复会话时按保存的配置重建模型适配器。
            owns_model_client: 是否由运行时负责关闭模型关联的 SDK 客户端。
            memory_store: 可选的全局记忆存储；省略时此底层构造函数不启用记忆。
            auto_memory: 是否为完整用户回合启动后台提取。
            extraction_model_factory: 可选的独立提取模型工厂；注入主模型或客户端时
                必须提供此工厂才运行后台提取，避免并发复用外部模型或意外发起网络调用。

        返回:
            None。初始化后的实例持有当前会话 ID、模型上下文及执行依赖。

        异常:
            ValueError: 新建会话要求开启思考，但当前适配器尚未接入思考开关；
                或工具注册表已经占用内建记忆、历史工具名。

        行为说明:
            1. 保存依赖和资源归属标记，初始化回合状态及上次执行结果。
            2. 使用注入的存储或创建存储，并注册只读历史工具；无 session_info 时保存新会话。
            3. 激活目标会话并加载上下文；此处不运行历史工具，也不请求模型。
            4. 构造失败时不自行关闭依赖；通过 create_runtime 创建时由工厂清理。
        """

        self.model = model
        self.connection = connection
        self.tool_registry = tool_registry
        self.logger = logger
        self.max_tool_rounds = max_tool_rounds
        self.thinking_enabled = thinking_enabled
        self.system_prompt = system_prompt
        self.messages: list[dict] = []
        self.last_result: AgentTurnResult | None = None
        self.last_input_retained = False
        self._closed = False
        self._responding = False
        self._storage_failed = False
        self._model_settings = model_settings
        self._model_client = model_client
        self._rebuild_model = rebuild_model
        self._owns_model_client = owns_model_client
        self.memory_store = memory_store
        self._memory_context = ContextBuilder(memory_store) if memory_store is not None else None
        self._memory_source: dict[str, str] | None = None
        self._memory_events: list[dict[str, Any]] = []
        self._extraction_model_factory = extraction_model_factory
        self.memory_extractor: MemoryExtractor | None = None
        if memory_store is not None:
            if {"remember", "read_memory"}.intersection(tool_registry.handlers):
                raise ValueError("工具注册表已占用内建记忆工具名")
            self.tool_registry = ToolRegistry(
                handlers={
                    **tool_registry.handlers,
                    "remember": self._remember,
                    "read_memory": memory_store.read_topic,
                },
                schemas=[*deepcopy(tool_registry.schemas), *deepcopy(PRIMARY_MEMORY_TOOL_SCHEMAS)],
                instructions=tool_registry.instructions,
            )
        self.recovered_turns = 0
        self.thinking_state_corrected = False
        if session_info is None and thinking_enabled and not _supports_thinking(model):
            raise ValueError("Kairos 尚未为当前模型接入思考开关，无法创建开启思考的会话")
        history_names = {"search_history", "read_session"}
        registered_names = set(self.tool_registry.handlers) | {
            schema.get("function", {}).get("name") for schema in self.tool_registry.schemas
        }
        if history_names.intersection(registered_names):
            raise ValueError("工具注册表已占用内建历史工具名")
        self.session_store = session_store or SessionStore(connection)
        self.history_reader = HistoryReader(self.session_store)
        self.tool_registry = ToolRegistry(
            handlers={
                **self.tool_registry.handlers,
                "search_history": self.history_reader.search_history,
                "read_session": self.history_reader.read_session,
            },
            schemas=[*deepcopy(self.tool_registry.schemas), *deepcopy(HISTORY_TOOL_SCHEMAS)],
            instructions=self.tool_registry.instructions,
        )
        info = session_info or self.session_store.create_session(
            system_prompt,
            provider=_model_provider(model),
            model=self.model_name,
            thinking_enabled=thinking_enabled,
        )
        self._activate_session(info, model=model, recover=session_info is not None)
        if (
            memory_store is not None
            and auto_memory
            and (extraction_model_factory is not None or (rebuild_model and model_client is None))
        ):
            self.memory_extractor = MemoryExtractor(memory_store)

    def _prepare_messages(self, messages: list[dict], *, turn_time: datetime) -> list[dict]:
        """组装每次模型请求的记忆、工具能力和本轮时间参考。

        作用:
            为新请求和工具续轮生成视图，使旧会话也使用当前工具规则及本轮时间。

        参数:
            messages: 当前会话的原始模型消息列表。
            turn_time: 本轮用户输入开始处理时捕获的本机时间，包含本地时区偏移。
                同一用户回合的所有请求共用此值。

        返回:
            独立的请求消息列表 (list[dict])，不会改写原始历史。

        行为说明:
            1. 记忆启用时先生成记忆视图，未启用时直接使用原始消息作为输入。
            2. 始终补充当前工具能力；业务模块卸载后，其请求期规则不再加入。
            3. 在请求副本的首条 system 中补充本轮日期、星期、时间和时区，
               不重新读取时钟，不向原始会话消息写入动态信息。
        """
        request = self._memory_context(messages) if self._memory_context is not None else messages
        request = prepare_tool_messages(request, self.tool_registry)
        weekday = "一二三四五六日"[turn_time.weekday()]
        time_context = (
            "[本轮时间参考]\n"
            f"本机当前时间：{turn_time.isoformat(timespec='seconds')}（星期{weekday}）。\n"
            f"本机时区：{turn_time.tzname()}（UTC 偏移 {turn_time.strftime('%z')}）。\n"
            "以上时间由运行时在本轮用户输入开始处理时提供，同一轮工具调用及后续模型请求保持一致。\n"
            "理解当前用户消息中的今天、明天、昨天等相对时间时，以此为参照；"
            "用户明确指定其他日期或时区时按用户要求处理。此参考不代表历史消息发生的时间。"
        )
        system = next(message for message in request if message.get("role") == "system")
        if isinstance(system["content"], list):
            system["content"].append({"type": "text", "text": time_context})
        else:
            system["content"] = f"{system['content']}\n\n{time_context}"
        return request

    def _remember(self, content: str) -> dict:
        """把当前用户明确要求保存的内容交给核心记忆存储。

        作用:
            作为主 Agent 的 remember 工具，绑定真实来源并保护调用边界。

        参数:
            content: 模型从用户明确保存要求中整理出的新核心内容。

        返回:
            存储层提供的保存或重复结果（dict）。

        异常:
            RuntimeError: 没有启用记忆，或当前没有活动用户回合。

        行为说明:
            1. 只允许活动用户回合调用，不让工具参数指定来源或写入路径。
            2. 绑定本轮用户消息 ID 后委托存储追加，语义查重由模型执行。
        """
        if self.memory_store is None or not self._responding or self._memory_source is None:
            raise RuntimeError("remember 只能在启用记忆的用户回合中调用")
        return self.memory_store.remember(content, source=self._memory_source)

    @property
    def memory_pending_count(self) -> int:
        """返回等待或正在执行的后台记忆任务数量。

        作用:
            供 CLI 退出提示和嵌入调用方判断后台工作是否已经结束。

        返回:
            未完成任务数量（int）；未启用提取时为零。

        行为说明:
            1. 只查询提取器状态，不访问用户会话数据库。
        """
        return self.memory_extractor.pending_count if self.memory_extractor is not None else 0

    def drain_memory_events(self) -> list[dict]:
        """取出尚未展示的后台记忆结果。

        作用:
            让主线程安全展示后台结果，后台线程不直接操作终端。

        返回:
            完成、无新增、冲突或失败事件（list[dict]），本次取出后不重复返回。

        行为说明:
            1. 取出运行时排队故障事件及提取器已完成事件，不修改原会话历史。
        """
        events, self._memory_events = self._memory_events, []
        if self.memory_extractor is not None:
            events.extend(self.memory_extractor.drain_events())
        return events

    def wait_for_memory(self, timeout: float | None = None) -> bool:
        """等待当前已提交的后台记忆工作结束。

        作用:
            提供正常收尾与离线测试的等待入口，不启动新的提取。

        参数:
            timeout: 最长等待秒数；None 表示等待完成。

        返回:
            是否已经全部完成（bool）；未启用提取时为 True。

        行为说明:
            1. 委托后台队列等待，任务结果仍通过 drain_memory_events 获取。
        """
        return self.memory_extractor is None or self.memory_extractor.wait(timeout)

    def _schedule_memory(self, source: dict[str, str], current_turn_start: int) -> None:
        """在完整用户回合结束后提交独立的记忆提取任务。

        作用:
            连接前台会话和后台提取，保留本轮边界与理解指代所需的前文。

        参数:
            source: 本轮真实用户消息来源。
            current_turn_start: 本轮用户消息在原历史中的位置。

        返回:
            None；没有提取器时不处理。

        行为说明:
            1. 固定来源、完整会话快照和本轮模型配置，不共享主线程 SQLite 连接。
            2. 使用注入工厂或独立创建模型；默认 DeepSeek 后台启用 JSON 模式，保留思考设置。
            3. 排队故障只形成后台失败事件，不将已经完成的用户回答改为失败。
        """
        if self.memory_extractor is None:
            return
        try:
            factory = self._extraction_model_factory
            if factory is None:
                settings = self._model_settings.model_copy(
                    update={"kairos_provider": _model_provider(self.model), "kairos_model": self.model_name}
                )
                thinking = self.thinking_enabled
                factory = partial(_build_extraction_model, settings, thinking_enabled=thinking)
            records = self.session_store.load_messages(source["session_id"])
            context_ids = [record.message_id for record in records if record.payload.get("role") != "system"]
            bound_source = dict(source)
            if context_ids:
                bound_source.update(context_start_message_id=context_ids[0], context_end_message_id=context_ids[-1])
            self.memory_extractor.submit(
                ExtractionJob(
                    source=bound_source,
                    messages=deepcopy(self.messages),
                    current_turn_start=current_turn_start,
                    model_factory=factory,
                )
            )
        except Exception as error:
            self._memory_events.append(
                {
                    "event": "memory_extraction",
                    "status": "failed",
                    "source": dict(source),
                    "core_additions": 0,
                    "error_code": type(error).__name__,
                    "topic_ids": [],
                }
            )

    @property
    def model_name(self) -> str:
        return str(getattr(self.model, "model", "unknown"))

    @property
    def supports_thinking(self) -> bool:
        """返回当前适配器是否已接入思考开关。

        作用:
            向 CLI 和设置入口提供与实际请求参数一致的支持状态。

        参数:
            无显式参数；读取当前激活的 model。

        返回:
            当前模型适配器是否允许切换思考（bool）。

        行为说明:
            1. 使用当前模型重新判断，不缓存上一个会话的能力。
        """

        return _supports_thinking(self.model)

    @property
    def session_id(self) -> str:
        """返回当前激活会话的稳定 ID。

        作用:
            向 CLI、记录器和会话操作提供当前对话身份，不读取数据库。

        参数:
            无显式参数；读取实例的 _session_id。

        返回:
            当前会话的 ID；恢复会话时沿用保存的原 ID。

        行为说明:
            1. 直接返回激活会话时保存的 ID，不生成新值，也不更新活动时间。
        """

        return self._session_id

    @property
    def session_info(self) -> SessionInfo:
        """读取当前会话的最新元信息。

        作用:
            为调用方提供数据库中的标题、设置和消息数，避免使用过期的内存概要。

        参数:
            无显式参数；使用当前 session_id 和 session_store。

        返回:
            当前会话的概要记录（SessionInfo）。

        行为说明:
            1. 检查运行时尚未关闭，再按当前 ID 查询会话，不修改存储内容。
        """

        self._ensure_open()
        return self.session_store.get_session(self.session_id)

    def _ensure_open(self) -> None:
        """检查运行时是否仍可使用其持有的资源。

        作用:
            作为读取和修改操作的前置检查，阻止已关闭实例继续访问数据库。

        参数:
            无显式参数；读取实例的 _closed 标记。

        返回:
            None，表示运行时尚未标记为关闭。

        异常:
            RuntimeError: _closed 为 True，显式拒绝继续使用已关闭的运行时。

        行为说明:
            1. 只检查本实例的关闭标记，不探测连接是否被外部代码关闭。
        """

        if self._closed:
            raise RuntimeError("AgentRuntime 已经关闭")

    def _ensure_idle(self) -> None:
        """检查运行时是否打开且没有正在执行的回合。

        作用:
            在切换会话、修改配置或释放资源前检查状态，避免影响当前回合。

        参数:
            无显式参数；读取实例的 _closed 和 _responding 标记。

        返回:
            None，表示本实例允许开始需要空闲状态的操作。

        异常:
            RuntimeError: _responding 为 True，当前回合仍在执行。

        行为说明:
            1. 先检查关闭状态，再检查回合执行标记；本函数不改变任何状态。
            2. 此检查依赖当前实例的标记，不提供跨线程或跨进程锁。
        """

        self._ensure_open()
        if self._responding:
            raise RuntimeError("当前回合仍在执行，请等待结束后再操作会话")

    def _activate_session(self, info: SessionInfo, *, model: Model, recover: bool) -> None:
        """加载目标会话的上下文并更新当前会话状态。

        作用:
            供初始化、新建和恢复操作共用，连接持久会话与运行时的内存消息。
            按需修补中断记录并更新活动时间，成功后清除上次结果和保存故障标记。

        参数:
            info: 已保存会话的元信息，提供 session_id 和 thinking_enabled。
            model: 即将激活的目标模型，不能用旧会话的模型判断目标开关。
            recover: 是否先修补中断回合；新会话为 False，恢复已有会话为 True。

        返回:
            None。实例改为持有目标会话的上下文、提示词及思考设置。

        异常:
            ValueError: 历史消息为空，或首条消息不是 role="system"，或其 content
                不是字符串，无法作为恢复后的系统提示词。

        行为说明:
            1. 按 recover 决定是否修补中断记录，再加载 in_context 消息。
               修补只补齐记录和状态，不重新执行工具。
            2. 校验系统消息及图片预算，按目标模型纠正旧的无效思考开关并更新活动时间。
               前面的修补可能已经提交，因此加载失败不代表数据库完全未改变。
            3. 存储操作成功后应用开关并一起切换模型与会话，重置上次结果和故障标记，
               并记录本次修补的回合数；调用方负责事先检查空闲状态。
        """

        recovered = self.session_store.recover_session(info.session_id) if recover else 0
        messages = [record.payload for record in self.session_store.load_messages(info.session_id)]
        if not messages or messages[0].get("role") != "system" or not isinstance(messages[0].get("content"), str):
            raise ValueError("会话缺少有效的 system 提示词，无法恢复")
        validate_image_budget(messages, ())
        thinking_corrected = info.thinking_enabled and not _supports_thinking(model)
        if thinking_corrected:
            info = self.session_store.update_thinking(info.session_id, False)
        self.session_store.touch_session(info.session_id)
        self._apply_thinking(info.thinking_enabled, model=model)
        self.model = model
        self._session_id = info.session_id
        self.messages = messages
        self.system_prompt = messages[0]["content"]
        self.last_result = None
        self.last_input_retained = False
        self.recovered_turns = recovered
        self.thinking_state_corrected = thinking_corrected
        self._storage_failed = False

    def new_session(self, title: str | None = None, system_prompt: str | None = None) -> SessionInfo:
        """创建并激活一个新的持久化会话。

        作用:
            响应新建对话或重置上下文操作，保存新会话及系统消息并切换运行时。
            旧会话历史与工具的业务数据继续保留。

        参数:
            title: 可选的手动标题；给出时必须是去除首尾空白后非空的字符串。
                省略时使用默认标题，接受第一条输入后由存储自动命名。
            system_prompt: 新会话的系统提示词；None 表示继承当前提示词。

        返回:
            已激活的新会话元信息（SessionInfo）。

        行为说明:
            1. 检查空闲状态，沿用当前协议、模型名称和思考设置创建会话。
            2. 不执行中断恢复，直接加载新会话并返回最新概要。
            3. 创建和激活是分开的操作；激活失败时，已创建的会话仍可能保存在库中。
        """

        self._ensure_idle()
        info = self.session_store.create_session(
            self.system_prompt if system_prompt is None else system_prompt,
            provider=_model_provider(self.model),
            model=self.model_name,
            thinking_enabled=self.thinking_enabled,
            title=title,
        )
        self._activate_session(info, model=self.model, recover=False)
        return self.session_info

    def list_sessions(self) -> list[SessionInfo]:
        """按最近活动顺序列出可见的历史会话。

        作用:
            为会话选择界面提供存储中的概要列表，不切换当前会话。

        参数:
            无显式参数；读取实例的 session_store，使用存储默认的 50 条上限。

        返回:
            按最近活动时间倒序排列的 SessionInfo 列表；无可见会话时返回空列表。

        行为说明:
            1. 检查运行时尚未关闭，再调用存储查询。
            2. 隐藏从未开始回合且没有手动标题的空会话，不更新活动时间。
        """

        self._ensure_open()
        return self.session_store.list_sessions()

    def rename_session(self, title: str) -> SessionInfo:
        """保存当前会话的手动标题。

        作用:
            接收 CLI 或其他调用方提供的标题，更新持久会话及最后活动时间。

        参数:
            title: 新标题，必须是去除首尾空白后非空的字符串；由存储层校验。

        返回:
            保存新标题后的会话元信息（SessionInfo）。

        行为说明:
            1. 检查空闲状态，再将当前会话 ID 和标题交给存储层。
            2. 存储层标记标题为手动设置，首条用户输入不再自动覆盖该标题。
        """

        self._ensure_idle()
        return self.session_store.rename_session(self.session_id, title)

    def resume_session(self, reference: str) -> SessionInfo:
        """恢复指定会话的历史、系统提示词和模型配置。

        作用:
            将 CLI 或调用方选择的已有会话激活到当前运行时，修补中断记录。
            必要时重建适配器；恢复只操作历史和配置，不重新执行历史工具。

        参数:
            reference: 目标会话的完整 ID 或可唯一匹配的非空前缀。
                前缀不匹配或匹配多个会话时，由存储层拒绝恢复。

        返回:
            恢复并激活后的会话元信息（SessionInfo）。

        异常:
            ValueError: 协议或模型名称与当前模型不同时，满足以下任一情况:
                - 不允许重建模型，或缺少重建所需的 model_settings;
                - 目标会话的协议不是 openai 或 anthropic，需要显式注入匹配模型;
                - 已注入 model_client，且目标会话需要切换协议。
            捕获的原异常（显式重新抛出）: 激活目标会话失败或被中断时，按客户端
                归属尝试关闭此次新建的客户端，再重新抛出原异常。

        行为说明:
            1. 检查空闲状态并读取会话；协议或模型变化时，按允许的重建策略
               准备适配器，认证信息仍来自当前配置，不从历史中恢复密钥。
            2. 激活目标会话。激活失败时，关闭此次新建且未使用注入客户端的
               适配器客户端；已经完成的数据库修补不会因此撤销。
            3. 激活过程统一替换模型并应用思考设置；仅在拥有关闭权时释放旧客户端。
               旧客户端关闭失败可能发生在切换已经完成之后。
        """

        self._ensure_idle()
        info = self.session_store.get_session(reference)
        model = self.model
        if (info.provider, info.model) != (_model_provider(model), self.model_name):
            if not self._rebuild_model or self._model_settings is None:
                raise ValueError("恢复会话需要相同协议和模型；请传入与该会话匹配的模型")
            if info.provider not in {"openai", "anthropic"}:
                raise ValueError("该会话使用自定义模型，请显式传入匹配的模型后恢复")
            if self._model_client is not None and info.provider != _model_provider(model):
                raise ValueError("注入的客户端不能跨协议切换，请使用对应协议的客户端重新恢复")
            saved_settings = self._model_settings.model_copy(
                update={"kairos_provider": info.provider, "kairos_model": info.model}
            )
            model = build_model(saved_settings, self._model_client, thinking_enabled=info.thinking_enabled)

        previous_model = self.model
        try:
            self._activate_session(info, model=model, recover=True)
        except (Exception, KeyboardInterrupt):
            if model is not previous_model and self._model_client is None:
                self._close_model_client(model)
            raise
        if model is not previous_model and self._owns_model_client:
            self._close_model_client(previous_model)
        return self.session_info

    def session_usage(self) -> SessionUsage:
        """从数据库读取当前会话的累计用量和已保存费用。

        作用:
            为回合结束、程序启动和会话切换提供相同统计来源，重启后不丢失累计值。

        参数:
            无显式参数；使用当前 session_id 读取步骤明细。

        返回:
            会话统计（SessionUsage）；新会话为零，缺失用量和费用保持未知标记。

        行为说明:
            1. 检查运行时尚未关闭，只读取当前会话的步骤，不加载消息正文或图片。
            2. 包含失败或中断回合已保存的明细，不混入其他会话和后台记忆提取。
        """
        self._ensure_open()
        return summarize_usage(self.session_store.usage_steps(self.session_id))

    def session_details(self) -> dict:
        """读取当前会话的完整关联记录并生成图片摘要。

        作用:
            为 ID 关系检查和历史审计提供展示数据，包含已退出模型上下文的消息。

        参数:
            无显式参数；按当前 session_id 读取 session_store。

        返回:
            包含 session、turns、steps、messages 和 tool_calls 的字典。
            图片数据替换为摘要，保留对象 ID 与关联字段。

        行为说明:
            1. 检查运行时尚未关闭，再加载五类对象及全部消息记录。
            2. 对结果生成图片脱敏副本，不修改数据库，也不改变模型上下文。
        """

        self._ensure_open()
        return redact_images(self.session_store.inspect(self.session_id))

    def respond(
        self,
        prompt: str,
        *,
        images: Sequence[ImageInput] = (),
        image_labels: Sequence[str] = (),
        on_event: EventLogger | None = None,
        cancel_token: CancellationToken | None = None,
        on_turn_start: Callable[[dict[str, str]], None] | None = None,
    ) -> AgentTurnResult:
        """处理一次用户输入并保存完整回合的执行记录。

        作用:
            接收文字及已加载的图片，在会话中建立 Turn 和用户消息，再驱动
            Agent 循环请求模型、执行工具和保存 Step，最终向调用方返回本轮结果。

        参数:
            prompt: 用户文字，必须为字符串；没有图片时还必须包含非空白内容。
            images: 已加载验证的 ImageInput 序列，新增后不得超过整个会话的图片预算。
            image_labels: 图片标签；省略时生成“图片 1”等标签。给出时必须与图片
                数量相同，每个标签均为非空白字符串且互不重复。
            on_event: 可选的实时事件回调，接收循环输出的正文增量和执行关联 ID。
                不接收原始请求、完整响应或工具载荷；实例 logger 另行接收完整观察事件。
            cancel_token: 可选的线程安全合作取消标记；启用时优先使用模型的流式接口，
                在模型与工具边界停止，不强制打断正在等待的网络或同步工具。
            on_turn_start: 可选同步关联回调，接收 session_id、turn_id、user_message_id 字典。
                Turn 已保存且模型、工具尚未执行时调用；必须确认关联持久化后返回。

        返回:
            完成回合的 AgentTurnResult，包含最终正文、消息副本、本轮用量、会话累计、
            工具结果及 session_id、turn_id、step_ids；同时保存到 last_result。
            回合失败或中断时不返回部分结果。

        异常:
            RuntimeError: 满足以下任一情况时显式抛出:
                - _responding 为 True，当前回合仍在执行;
                - _storage_failed 为 True，之前的保存故障尚未通过恢复清除;
                - 执行失败后保存回合结束状态再次遇到 sqlite3.Error;
                - 捕获的 Exception 经 sanitize_error 处理后文本发生变化，需要
                  改为抛出不含原敏感内容的错误。
            ValueError: 满足以下任一情况时显式抛出:
                - prompt 不是字符串，或文字为空白且没有图片;
                - 标签与图片数量不符，或有标签不是非空白字符串;
                - 图片标签重复。
            捕获的原异常（显式重新抛出）: 执行失败或被中断时，完成本地输入保留
                处理并尝试保存回合结束状态；未转为上述 RuntimeError 时重抛原异常。

        行为说明:
            1. 拒绝重复执行，清空上次结果并检查运行状态、输入、标签及图片预算。
               这些准入检查发生在写入 Turn 之前；图片随后组装为完整用户消息。
            2. 捕获一次本机当前时间与时区，供本轮所有模型请求共用；
               在存储中创建 Turn 和输入消息，再加入内存上下文并启动记录器。
               关联回调成功返回后才允许模型和工具执行；回调失败沿用回合失败收尾。
               循环负责在实际边界保存步骤、助手消息及工具调用和结果。
            3. 成功时保存回合完成状态，并汇总事件生成返回值。执行失败时，若
               本轮尚无完整助手消息，则撤出上下文中的输入，数据库审计记录保留；
               已有助手消息和工具记录则继续保留，避免遗忘已发生的副作用。
            4. 循环或回合收尾阶段发现 SQLite 保存故障后标记 _storage_failed，
               当前会话需显式恢复后才能继续提交；begin_turn 失败则直接向外传播。
               无论循环如何退出，finally 都会清除正在执行标记。
            5. 已取消的请求在创建 Turn 前停止；执行中取消使用 KeyboardInterrupt 子类，
               沿用 cancelled 状态及工具记录保留规则。作用域退出后不影响后续请求和后台记忆。
        """

        if self._responding:
            raise RuntimeError("当前回合仍在执行，请等待结束后再提交输入")
        self.last_input_retained = False
        self.last_result = None
        self._ensure_idle()
        if self._storage_failed:
            raise RuntimeError("会话保存曾失败，请处理存储问题并 /resume 当前会话后继续")
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        if not isinstance(prompt, str) or (not prompt.strip() and not images):
            raise ValueError("用户消息不能为空")
        images = tuple(images)
        labels = tuple(image_labels) or tuple(f"图片 {index}" for index in range(1, len(images) + 1))
        if len(labels) != len(images) or any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ValueError("图片标签必须与附件一一对应")
        if len(set(labels)) != len(labels):
            raise ValueError("图片标签不能重复")
        self.check_image_budget(images)
        content: str | list[dict] = prompt
        if images:
            content = [{"type": "text", "text": prompt if prompt.strip() else "请描述这些图片。"}]
            for label, image in zip(labels, images, strict=True):
                content.append({"type": "text", "text": f"[{label}：{image.name}]"})
                content.append(image.to_block())

        original_length = len(self.messages)
        user_message = {"role": "user", "content": content}
        turn_time = datetime.now().astimezone()
        turn = self.session_store.begin_turn(self.session_id, user_message, title_hint=prompt.strip() or "图片对话")
        self.messages.append(user_message)
        self.last_input_retained = True
        self._responding = True
        source = {"session_id": self.session_id, "turn_id": turn.turn_id, "message_id": turn.user_message_id}
        self._memory_source = source
        recorder = TurnRecorder(self.session_store, self.session_id, turn.turn_id, self.model_name)
        events: list[dict[str, Any]] = []

        def record(event: dict[str, Any]) -> None:
            events.append(event)
            if self.logger:
                self.logger(event)

        try:
            with cancellation_scope(cancel_token):
                if on_turn_start is not None:
                    on_turn_start(
                        {
                            "session_id": self.session_id,
                            "turn_id": turn.turn_id,
                            "user_message_id": turn.user_message_id,
                        }
                    )
                raise_if_cancelled()
                content, messages = run_agent_turn(
                    self.messages,
                    self.model,
                    self.tool_registry.handlers,
                    self.tool_registry.schemas,
                    max_tool_rounds=self.max_tool_rounds,
                    logger=record,
                    on_event=on_event,
                    recorder=recorder,
                    prepare_messages=partial(self._prepare_messages, turn_time=turn_time),
                )
            self.session_store.finish_turn(turn.turn_id, "completed", "final_answer")
        except (Exception, KeyboardInterrupt) as error:
            safe_error = sanitize_error(error, self.messages)
            # 尚未保存完整模型消息时撤回本轮输入。若已有工具执行，则保留其
            # 完整历史，不能让下次请求因遗忘这些副作用而重复执行。
            if len(self.messages) == original_length + 1:
                del self.messages[original_length:]
                self.last_input_retained = False
            end_reason = "user_interrupt" if isinstance(error, KeyboardInterrupt) else "error"
            if isinstance(error, AgentLoopLimitError):
                end_reason = "max_tool_rounds"
            if isinstance(error, sqlite3.Error) or recorder.storage_failed:
                self._storage_failed = True
            try:
                self.session_store.finish_turn(
                    turn.turn_id,
                    "cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                    end_reason,
                    retain_input=self.last_input_retained,
                    error=safe_error,
                )
            except sqlite3.Error:
                self._storage_failed = True
                raise RuntimeError("会话保存失败，本轮已停止；请处理存储问题并恢复当前会话后继续") from None
            if isinstance(error, Exception) and safe_error != str(error):
                raise RuntimeError(safe_error) from None
            raise
        finally:
            self._responding = False
            self._memory_source = None

        result = AgentTurnResult(
            content=content,
            messages=deepcopy(messages),
            usage=_aggregate_usage(events),
            finish_reason=_finish_reason(events),
            events=events,
            tool_results=_tool_results(events),
            session_id=self.session_id,
            turn_id=turn.turn_id,
            step_ids=tuple(recorder.step_ids),
            last_request_usage=_last_request_usage(events),
            session_usage=self.session_usage(),
        )
        self.last_result = result
        self._schedule_memory(source, original_length)
        return result

    def reset(self, system_prompt: str | None = None) -> None:
        """开启新会话；沿用模型、工具和连接，保留旧会话供恢复。"""

        self.new_session(system_prompt=system_prompt)

    def check_image_budget(self, images: Sequence[ImageInput]) -> None:
        """在草稿或用户消息进入历史前检查整个会话的图片预算。"""

        validate_image_budget(self.messages, images)

    def set_thinking(self, enabled: bool) -> None:
        """校验并保存当前会话的思考开关。

        作用:
            作为手动切换入口，保证保存状态、界面状态与实际模型请求参数一致。

        参数:
            enabled: 要保存的布尔开关。

        返回:
            None。保存成功后更新内存模型的请求参数。

        异常:
            ValueError: 参数不是 bool，或当前适配器尚未接入思考开关。

        行为说明:
            1. 检查运行时空闲、参数类型及支持能力；不支持时不修改任何设置。
            2. 先保存，再更新模型参数；存储失败时保留内存中的原状态。
        """

        self._ensure_idle()
        if type(enabled) is not bool:
            raise ValueError("思考开关必须是布尔值")
        if not self.supports_thinking:
            raise ValueError("Kairos 尚未为当前模型接入思考开关")
        self.session_store.update_thinking(self.session_id, enabled)
        self._apply_thinking(enabled, model=self.model)

    def _apply_thinking(self, enabled: bool, *, model: Model) -> None:
        """将思考开关应用到运行时及支持它的内存模型。

        作用:
            供初始化、恢复和设置更新共用；持久化由调用方负责，本函数不写库。

        参数:
            enabled: 要应用的思考开关；调用方按布尔值传入，本函数不单独校验类型。
            model: 要应用参数的目标模型；恢复时可以与当前 self.model 不同。

        返回:
            None。更新实例的 thinking_enabled，并按模型类型更新请求附加参数。

        异常:
            ValueError: 要求开启思考，但目标适配器尚未接入思考开关。

        行为说明:
            1. 检查目标能力；不允许保存无法落实到请求的开启状态。
            2. 已接入时仅覆盖 extra_body 的 thinking 字段，保留其他请求选项。
            3. 更新内存开关；未接入的模型只允许保持关闭，不构造推测的协议参数。
        """

        supported = _supports_thinking(model)
        if enabled and not supported:
            raise ValueError("Kairos 尚未为当前模型接入思考开关")
        if supported:
            model.extra_body = {
                **(model.extra_body or {}),
                "thinking": {"type": "enabled" if enabled else "disabled"},
            }
        self.thinking_enabled = enabled

    def history(self) -> list[dict]:
        """生成当前模型上下文的图片摘要副本。

        作用:
            为 CLI 查看当前历史提供展示数据，原始上下文继续保留完整图片供模型使用。

        参数:
            无显式参数；读取实例当前的 messages 列表。

        返回:
            与当前上下文对应的消息列表副本，图片数据替换为摘要；空上下文返回空列表。

        行为说明:
            1. 递归生成消息的脱敏副本，不读取数据库、不修改原消息。
            2. 已退出上下文的审计消息不在此列表中，需要通过 session_details 查看。
        """

        return redact_images(self.messages)

    def close(self) -> None:
        """完成后台任务并关闭数据库及自有模型客户端。

        作用:
            在调用方结束使用运行时或退出上下文管理器时释放资源，持久会话仍保留。

        参数:
            无显式参数；读取关闭、执行和客户端归属标记，使用当前连接及模型。

        返回:
            None。已成功关闭的实例再次调用时直接返回。

        行为说明:
            1. 尚未关闭时检查前台空闲状态，等待已排队的后台提取完成。
            2. 关闭自有模型客户端，并在 finally 中关闭数据库连接。
            3. 连接关闭成功后设置 _closed；若连接关闭自身抛出异常，该标记不会更新。
               外部注入客户端通常由调用方管理，其关闭权由创建时的标记决定。
        """

        if not self._closed:
            self._ensure_idle()
            try:
                if self.memory_extractor is not None:
                    self.memory_extractor.close()
            finally:
                try:
                    if self._owns_model_client:
                        self._close_model_client(self.model)
                finally:
                    self.connection.close()
                    self._closed = True

    @staticmethod
    def _close_model_client(model: Model) -> None:
        """调用模型关联客户端提供的关闭方法。

        作用:
            供运行时关闭、模型替换和工厂失败清理共用，只负责执行资源释放动作。

        参数:
            model: 待清理的模型对象；由调用方事先确认有权关闭其客户端。

        返回:
            None。没有 client、没有 close 或 close 不可调用时也直接返回。

        行为说明:
            1. 读取 model.client.close，存在且可调用时执行，不另行判断资源归属。
            2. 忽略关闭方法的返回值，不修改运行时的关闭标记。
        """

        close = getattr(getattr(model, "client", None), "close", None)
        if callable(close):
            close()

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
    session_id: str | None = None,
    resume_latest: bool = False,
    memory_dir: str | Path | None = None,
    memory_enabled: bool = True,
    auto_memory: bool = True,
    extraction_model_factory: Callable[[], Model] | None = None,
    tool_factories: Sequence[str] | None = None,
) -> AgentRuntime:
    """创建并返回已激活会话的完整运行时。

    作用:
        作为 CLI 和测试的组装入口，初始化数据库、会话存储、工具及模型依赖。
        新建或恢复会话后交由 AgentRuntime 执行，初始化失败时尝试释放已创建资源。

    参数:
        settings: 当前模型及认证配置；未注入 model 且未给出时加载默认 Settings。
        client: 可选的外部 SDK 客户端，供工厂创建的模型适配器使用；关闭权保留给
            调用方。显式传入 model 时，其实际客户端由该模型自行持有。
        model: 可选的现成模型；省略时由工厂创建。恢复时注入模型的协议与名称
            必须匹配保存的会话，工厂不会自动替换注入模型。
        db_path: 数据库路径；省略时使用当前用户 ~/.kairos/kairos.sqlite3。
        log_path: 可选的 JSONL 观察日志路径；省略时不设置事件日志器。
        system_prompt: 新会话的系统提示词；恢复时采用保存的提示词。
        thinking_enabled: 新会话的思考开关；恢复时采用保存的开关。
        session_id: 要恢复的完整会话 ID 或唯一前缀，与 resume_latest 互斥。
        resume_latest: 是否恢复最近活动的可见会话；没有历史时抛出异常。
        memory_dir: 全局记忆目录；省略时使用 ~/.kairos/memory，测试可传临时路径。
        memory_enabled: 是否注册记忆工具并组装记忆上下文；默认开启。
        auto_memory: 是否启用每轮后台提取；默认开启，测试可关闭。
        extraction_model_factory: 可选的独立提取模型工厂。应用自行创建主模型时
            默认根据本轮配置独立创建提取模型；注入主模型或客户端时须显式提供。
        tool_factories: 本地工具工厂路径；None 读取配置，空序列关闭可选业务工具，
            非空序列显式替换配置。记忆与历史工具仍由其原有入口管理。

    返回:
        已激活新会话或指定历史的 AgentRuntime；调用方使用结束后应调用 close。

    异常:
        ValueError: 满足以下任一情况时显式抛出:
            - session_id 与 resume_latest 同时指定;
            - 由工厂管理模型时，目标会话协议不是 openai 或 anthropic;
            - 显式注入的模型与目标会话保存的协议或模型名称不匹配。
        捕获的原异常（显式重新抛出）: 打开数据库后的初始化失败或被中断时，
            依次尝试关闭工厂自有客户端和连接，再重新抛出原异常。

    行为说明:
        1. 检查恢复选项，准备配置并打开数据库，再创建存储及查找目标会话。
        2. 恢复时使用会话保存的协议、模型和思考设置；认证仍来自当前配置。
           注入模型则只检查兼容性，不按历史重建。
        3. 组装模型、日志和工具后构造运行时。只有工厂自建且未注入 client 的
           模型客户端才由运行时拥有关闭权；数据库连接始终交由运行时关闭。
        4. 打开连接后初始化失败时，依次尝试关闭自建客户端和连接，再传播异常。
           清理不回滚此前已经提交的初始化或会话写入；客户端关闭若抛出异常，
           当前实现也会中断后续连接清理。
    """

    if session_id is not None and resume_latest:
        raise ValueError("指定会话与继续最近会话不能同时使用")
    managed_model = model is None
    if managed_model:
        settings = settings or Settings()
    database_path = db_path if db_path is not None else config.get_app_dir() / "kairos.sqlite3"
    connection = open_database(database_path)
    runtime_model = None
    try:
        if tool_factories is None:
            tool_factories = (settings if settings is not None else ToolSettings()).kairos_tool_factories
        registry = load_tool_registry(connection, tool_factories)
        store = SessionStore(connection)
        info = store.get_session(session_id) if session_id is not None else None
        if resume_latest:
            info = store.latest_session()
        runtime_settings = settings
        if info is not None:
            thinking_enabled = info.thinking_enabled
            if managed_model:
                if info.provider not in {"openai", "anthropic"}:
                    raise ValueError("该会话使用自定义模型，请显式传入匹配的模型后恢复")
                runtime_settings = settings.model_copy(
                    update={"kairos_provider": info.provider, "kairos_model": info.model}
                )
            elif (info.provider, info.model) != (_model_provider(model), str(getattr(model, "model", "unknown"))):
                raise ValueError("恢复会话需要相同协议和模型；请传入与该会话匹配的模型")
        runtime_model = model or build_model(runtime_settings, client=client, thinking_enabled=thinking_enabled)
        logger = JsonlLogger(log_path) if log_path else None
        memory_store = (
            MemoryStore(Path(memory_dir) if memory_dir is not None else config.get_app_dir() / "memory")
            if memory_enabled
            else None
        )
        return AgentRuntime(
            model=runtime_model,
            connection=connection,
            tool_registry=registry,
            system_prompt=system_prompt,
            logger=logger,
            thinking_enabled=thinking_enabled,
            session_store=store,
            session_info=info,
            model_settings=settings,
            model_client=client,
            rebuild_model=managed_model,
            owns_model_client=managed_model and client is None,
            memory_store=memory_store,
            auto_memory=auto_memory,
            extraction_model_factory=extraction_model_factory,
        )
    except (Exception, KeyboardInterrupt):
        if runtime_model is not None and managed_model and client is None:
            AgentRuntime._close_model_client(runtime_model)
        connection.close()
        raise
