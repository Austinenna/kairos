"""一个会话：把历史、模型、工具和存储接在一起，respond() 处理一次用户输入。"""

from collections.abc import Callable
from datetime import datetime
from functools import partial

from .agent import TurnResult, run_turn
from .agent.context import build_request
from .config import Settings
from .models import Usage, build_model
from .models.catalog import estimate_cost, model_info
from .storage import Session, SessionStore, SessionTotals

DEFAULT_SYSTEM_PROMPT = """你是 Kairos，用户的个人助手。回答简洁直接，使用中文。
需要执行操作时调用对应工具，工具返回结果后再说明结果；不需要工具时直接回答。"""


class Conversation:
    def __init__(
        self,
        store: SessionStore,
        session: Session,
        settings: Settings,
        tools: list[dict],
        handlers: dict[str, Callable],
    ):
        """载入已有会话。新建会话用 Conversation.create()。"""
        self.store = store
        self.session = session
        self.settings = settings
        self.tools = tools
        self.handlers = handlers
        self.messages = store.load_messages(session.id)
        self.model = self._build_model()

    @classmethod
    def create(
        cls,
        store: SessionStore,
        settings: Settings,
        tools: list[dict],
        handlers: dict[str, Callable],
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> "Conversation":
        """用当前配置的协议和模型新建一个会话。"""
        session = store.create(
            provider=settings.kairos_provider, model=settings.kairos_model, system_prompt=system_prompt
        )
        return cls(store, session, settings, tools, handlers)

    def respond(
        self,
        prompt: str,
        on_text: Callable[[str], None] | None = None,
        on_message: Callable[[dict], None] | None = None,
    ) -> TurnResult:
        """处理一次用户输入：保存用户消息，运行 Agent 循环，循环中的每条消息都即时保存。

        on_text 接收正文片段；on_message 在每条新消息保存后调用，界面可以用它显示工具调用。
        """

        def save_and_notify(message: dict, usage: Usage | None) -> None:
            self._save(message, usage)
            if on_message:
                on_message(message)

        self._save({"role": "user", "content": prompt})
        self.messages.append({"role": "user", "content": prompt})
        # 同一轮里多次请求模型用同一个时间，保证这些请求的前缀相同、能命中缓存。
        prepare = partial(build_request, now=datetime.now().astimezone())
        return run_turn(
            self.messages,
            self.model,
            self.tools,
            self.handlers,
            prepare=prepare,
            on_text=on_text,
            on_message=save_and_notify,
        )

    def set_thinking(self, enabled: bool) -> bool:
        """切换思考模式。返回 False 表示当前模型不支持开关。"""
        if not model_info(self.session.model).thinking_toggle:
            return False
        self.store.set_thinking(self.session.id, enabled)
        self.session.thinking = enabled
        self.model = self._build_model()
        return True

    def rename(self, title: str) -> None:
        self.store.rename(self.session.id, title)
        self.session.title = title

    def totals(self) -> SessionTotals:
        return self.store.totals(self.session.id)

    def _save(self, message: dict, usage: Usage | None = None) -> None:
        cost = estimate_cost(self.session.model, usage) if usage else None
        self.store.add_message(self.session.id, message, usage, cost)

    def _build_model(self):
        """按会话保存的协议和模型创建适配器；密钥和地址仍用当前配置。"""
        settings = self.settings.model_copy(
            update={"kairos_provider": self.session.provider, "kairos_model": self.session.model}
        )
        return build_model(settings, thinking=self.session.thinking)
