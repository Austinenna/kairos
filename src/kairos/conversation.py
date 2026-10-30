"""一个会话：持有历史和模型，respond() 处理一次用户输入。共享的依赖都来自 Services。"""

from collections.abc import Callable
from datetime import datetime

from .agent import TurnResult, run_turn
from .agent.context import build_request
from .compaction.budget import compact_threshold
from .compaction.compact import CompactResult, compact, context_tokens, load_context
from .files.text import TextFiles
from .images.prepare import Image
from .images.refs import expand_image_refs
from .models import Usage, build_model
from .models.catalog import estimate_cost, model_info
from .services import Services
from .storage import Session, SessionTotals
from .tools import Toolset
from .tools.files import file_tools

DEFAULT_SYSTEM_PROMPT = """你是 Kairos，用户的个人助手。回答简洁直接，使用中文。
需要执行操作时调用对应工具，工具返回结果后再说明结果；不需要工具时直接回答。"""


class Conversation:
    def __init__(self, services: Services, session: Session):
        """载入已有会话。新建会话用 Conversation.create()。"""
        self.services = services
        self.session = session
        self.messages = load_context(services.sessions, session.id)
        self.model = self._build_model()
        # 文件工具按会话创建：先读后改检查要知道“这个会话”读过哪些文件。重新载入会话后需要重新读。
        self.file_tools = file_tools(TextFiles(services.file_policy))

    @classmethod
    def create(
        cls, services: Services, channel: str = "terminal", system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ) -> "Conversation":
        """用当前配置的协议和模型新建一个会话。channel 是创建它的入口：terminal 或 weixin。"""
        settings = services.settings
        session = services.sessions.create(
            provider=settings.kairos_provider,
            model=settings.kairos_model,
            system_prompt=system_prompt,
            channel=channel,
        )
        return cls(services, session)

    def respond(
        self,
        prompt: str,
        images: list[Image] | None = None,
        image_ids: list[int] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_message: Callable[[dict], None] | None = None,
        extra_tools: Toolset | None = None,
        extract_memory: bool = True,
        on_notice: Callable[[str], None] | None = None,
    ) -> TurnResult:
        """处理一次用户输入。

        1. 上下文超过压缩阈值时，先把较早的历史压缩成摘要（见 _auto_compact）。
        2. 保存用户消息，运行 Agent 循环，循环中的每条新消息都即时保存。
        3. 正常结束后，把这一轮交给后台提取记忆。

        images 是随这条消息一起发送的图片，原样存进历史（终端贴图）；
        image_ids 是已经归档的图片编号，历史里只存引用，请求前再展开（微信发图）；
        on_text 接收正文片段；on_message 在每条新消息保存后调用，界面可以用它显示工具调用。
        extra_tools 是只在这一轮可用的工具（定时任务的 skip_push）；extract_memory=False 时不做后台记忆提取。
        on_notice 接收给用户看的提示（压缩结果），不传就不提示（微信）。
        """

        def save_and_notify(message: dict, usage: Usage | None) -> None:
            self._save(message, usage)
            if on_message:
                on_message(message)

        # 重新载入历史：同一个会话可能刚在另一个入口（终端或微信）里有了新消息。
        self.messages = load_context(self.services.sessions, self.session.id)
        blocks = [image.to_block() for image in images or []]
        blocks += [{"type": "image_ref", "image_id": image_id} for image_id in image_ids or []]
        # 有图片时 content 是内容块列表：先文字，再逐张图片。
        content = [{"type": "text", "text": prompt or "请看这些图片。"}, *blocks] if blocks else prompt
        user = {"role": "user", "content": content}
        self._auto_compact(user, on_notice or (lambda notice: None))
        self._save(user)
        self.messages.append(user)
        tools = Toolset()
        tools.add(self.services.tools)
        tools.add(self.file_tools)
        if extra_tools:
            tools.add(extra_tools)
        # 时间和记忆在一轮开始时取一次，这一轮里多次请求模型都用同一份，前缀相同才能命中缓存。
        now, memory = datetime.now().astimezone(), self.services.memory.snapshot()
        rules = "\n\n".join(tools.instructions)

        def prepare(messages: list[dict]) -> list[dict]:
            view = expand_image_refs(messages, self.services.images)
            return build_request(view, now=now, rules=rules, memory=memory)

        result = run_turn(
            self.messages,
            self.model,
            tools.schemas,
            tools.handlers,
            prepare=prepare,
            on_text=on_text,
            on_message=save_and_notify,
        )
        if self.services.extractor and extract_memory:
            self.services.extractor.submit(self.messages)
        return result

    def compact(self, focus: str = "", tokens_before: int | None = None) -> CompactResult | None:
        """把较早的历史压缩成摘要（/compact 和自动压缩共用）。focus 是要求摘要重点关注的内容。

        返回 None 表示没有可以压缩的消息：保留最近约 20K 原文之后，前面已经没有内容了。
        """
        model = self._build_model(thinking=False)  # 写摘要不需要思考
        result = compact(self.services.sessions, self.session.id, model, tokens_before or self.context_tokens(), focus)
        if result:
            self.messages = load_context(self.services.sessions, self.session.id)
            # 摘要代替了原文，模型对读过的文件只剩模糊印象：清空“已读”记录，改文件前必须重新读。
            self.file_tools = file_tools(TextFiles(self.services.file_policy))
        return result

    def compact_threshold(self) -> int | None:
        """超过这个 token 数就自动压缩；模型窗口未知时为 None，不自动压缩。"""
        window = model_info(self.session.model).context_window
        return compact_threshold(window, self.services.settings.kairos_auto_compact_window_percent)

    def context_tokens(self, extra: list[dict] | None = None) -> int:
        """当前上下文的大小（部分是估算值）。extra 是还没保存的消息。"""
        return context_tokens(self.services.sessions, self.session.id, extra or [])

    def append_assistant(self, text: str) -> None:
        """把程序发出的一条消息（例如定时推送）作为助手消息存进历史，用户接着回复时模型能看到。"""
        message = {"role": "assistant", "content": text}
        self._save(message)
        self.messages.append(message)

    def set_thinking(self, enabled: bool) -> bool:
        """切换思考模式。返回 False 表示当前模型不支持开关。"""
        if not model_info(self.session.model).thinking_toggle:
            return False
        self.services.sessions.set_thinking(self.session.id, enabled)
        self.session.thinking = enabled
        self.model = self._build_model()
        return True

    def rename(self, title: str) -> None:
        self.services.sessions.rename(self.session.id, title)
        self.session.title = title

    def totals(self) -> SessionTotals:
        return self.services.sessions.totals(self.session.id)

    def _save(self, message: dict, usage: Usage | None = None) -> None:
        cost = estimate_cost(self.session.model, usage) if usage else None
        self.services.sessions.add_message(self.session.id, message, usage, cost)

    def _auto_compact(self, user: dict, notify: Callable[[str], None]) -> None:
        """一轮开始时检查：加上这条用户消息后超过阈值，就先压缩。在这里检查而不是每次请求前，
        是因为一轮里多次请求共用同一个前缀，中途压缩会让这一轮的缓存失效。"""
        threshold = self.compact_threshold()
        tokens = self.context_tokens([user])
        if threshold is None or tokens <= threshold:
            return
        notify("上下文超过压缩阈值，正在把较早的对话压缩成摘要……")
        try:
            result = self.compact(tokens_before=tokens)
        except Exception as error:  # 压缩失败不影响这一轮：历史不变，照常用完整历史请求，下一轮开始时会再试
            notify(f"上下文压缩失败，这一轮仍用完整历史：{type(error).__name__}: {error}")
            return
        if result:
            notify(result.notice())

    def _build_model(self, thinking: bool | None = None):
        """按会话保存的协议和模型创建适配器；密钥和地址仍用当前配置。thinking 不传时用会话的设置。"""
        settings = self.services.settings.model_copy(
            update={"kairos_provider": self.session.provider, "kairos_model": self.session.model}
        )
        return build_model(settings, thinking=self.session.thinking if thinking is None else thinking)
