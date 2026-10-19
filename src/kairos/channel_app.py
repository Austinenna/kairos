"""协调聊天通道与统一应用客户端，保存输入、会话绑定和投递结果。"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, uuid5

from .channels.base import ChannelAdapter, InboundMessage, ReceiveBatch, ReplyTarget, TransportError
from .channels.store import INTERRUPTED_REPLY, ChannelStore, StoredMessage

if TYPE_CHECKING:
    from .application import RuntimeClient

HELP_TEXT = """直接发送文字即可与 Kairos 对话。
/new [标题]：开始新会话，保留旧记录和全局记忆
/session：查看当前会话
/retry：重发当前会话最近一次已生成的回答，不重新执行任务；可能重复收到
/help：查看此说明
当前支持本人私聊文字，图片、语音和文件尚未接入。"""
FAILURE_TEXT = "本轮未能正常完成。为避免重复操作，Kairos 没有自动重试，请检查会话记录后再决定下一步。"
UNSUPPORTED_TEXT = "目前微信入口只支持纯文字，请把问题作为文字消息发送。"


def split_reply(text: str, byte_limit: int = 1800) -> list[str]:
    """将完整回复分成有序的 UTF-8 文字段落。

    作用:
        给通道保存稳定的分段结果，后续投递不再生成或改写答案。
    参数:
        text: 完整助手回复。
        byte_limit: 每段连同编号的项目字节预算，不代表平台公布的限制。
    返回:
        不拆开 Unicode 字符的文字片段列表；多段带顺序编号。
    异常:
        ValueError: 字节预算小于 64，或回复不是非空字符串。
    行为说明:
        1. 为段号预留空间，尽量在后半段的换行处切分，不删除正文。
        2. 单段不加编号，多段按生成顺序编号。
    """
    if byte_limit < 64 or not isinstance(text, str) or not text:
        raise ValueError("回复必须为非空文字，分段预算至少为 64 字节")
    if len(text.encode("utf-8")) <= byte_limit:
        return [text]
    budget = byte_limit - 32
    chunks, current, size = [], "", 0
    for character in text:
        length = len(character.encode("utf-8"))
        if size + length > budget:
            newline = current.rfind("\n")
            cut = newline + 1 if newline >= len(current) // 2 else len(current)
            chunks.append(current[:cut])
            current = current[cut:]
            size = len(current.encode("utf-8"))
        current += character
        size += length
    if current:
        chunks.append(current)
    return [f"（{index}/{len(chunks)}）\n{chunk}" for index, chunk in enumerate(chunks, 1)]


@dataclass(frozen=True)
class PreparedReply:
    """保存一次输入的最终展示内容及其来源类别。"""

    text: str
    kind: str
    turn_id: str | None = None
    binding_session_id: str | None = None


class ReplyDeliveryError(TransportError):
    """表示答案已保存但发送失败，协调器可以继续接收用户的显式重发命令。"""


class ChannelApp:
    """在处理线程内使用通道存储和应用客户端，不创建或拥有业务 Runtime。"""

    def __init__(
        self,
        store: ChannelStore,
        adapter: ChannelAdapter,
        client_factory: Callable[..., RuntimeClient],
        *,
        report: Callable[[str], None] = print,
        reply_byte_limit: int = 1800,
        stop_event: threading.Event | None = None,
    ) -> None:
        """绑定通道依赖，暂不创建模型会话或连接网络。

        作用:
            把接收、执行和投递分开，便于使用临时数据库及假适配器验收。
        参数:
            store: 当前线程持有的独立通道存储，由调用方负责关闭。
            adapter: 负责授权范围内收发的同步适配器。
            client_factory: 接受 source、session_id 的应用客户端工厂，生产传 service.client。
            report: 仅接收无正文、无凭据的状态文字。
            reply_byte_limit: 保存出站片段时采用的项目字节预算。
            stop_event: 可选共享停止信号，不跨线程关闭 SQLite 或应用 Runtime。
        返回:
            无返回值。
        行为说明:
            1. 按稳定外部对话创建轻量客户端；所有业务执行由应用服务调度。
            2. 调用方负责通道运行锁与当前线程内的存储生命周期。
        """
        self.store = store
        self.adapter = adapter
        self.client_factory = client_factory
        self.report = report
        self.reply_byte_limit = reply_byte_limit
        self.clients: dict[str, RuntimeClient] = {}
        self.stop_event = stop_event or threading.Event()
        self._started = False

    def start(self) -> None:
        """恢复通道执行状态，但不重新执行旧任务。

        作用:
            为新进程识别上次处理或发送时中断的输入。
        返回:
            None。
        行为说明:
            1. 每个协调器只恢复一次；中断输入产生通知，未知投递不会自动重发。
        """
        if not self._started:
            recovered = self.store.recover(self.adapter.channel, self.adapter.account_id)
            self._started = True
            if recovered:
                self.report("已发现上次中断的通道记录；没有重新执行模型或工具。")

    def poll_once(self) -> int:
        """读取并原子保存一批获准输入和平台接收进度。

        作用:
            先保存输入再推进游标，避免模型执行速度影响已收到消息的保存。
        返回:
            新插入的输入数量 (int)，不含重复消息。
        行为说明:
            1. 通道层再次核对账号及本人身份；未授权输入不进入数据库或模型。
            2. checkpoint 由适配器解释，协调器只把它随消息一起保存。
        """
        self.start()
        checkpoint = self.store.checkpoint(self.adapter.channel, self.adapter.account_id) or {}
        batch = self.adapter.poll(checkpoint)
        messages = tuple(
            message
            for message in batch.messages
            if message.channel == self.adapter.channel
            and message.account_id == self.adapter.account_id
            and message.sender_id == self.adapter.owner_id
        )
        return self.store.accept_batch(
            self.adapter.channel, self.adapter.account_id, ReceiveBatch(messages, batch.checkpoint)
        )

    @staticmethod
    def source_for(message: InboundMessage) -> str:
        """构造不随线程或重启变化的应用来源标识。

        作用:
            把同一外部账号与对话固定到同一客户端身份，避免跨入口混用请求。
        参数:
            message: 已获准的外部消息。
        返回:
            含通道名与身份摘要的稳定字符串 (str)。
        行为说明:
            1. 不包含消息正文、凭据或回复令牌。
        """
        identity = json.dumps([message.channel, message.account_id, message.conversation_id], ensure_ascii=True)
        return f"channel:{message.channel}:{hashlib.sha256(identity.encode()).hexdigest()}"

    def _select_session(self, message: InboundMessage, *, session_id: str | None = None) -> RuntimeClient:
        """按平台对话的持久绑定选择 Kairos 会话。

        作用:
            为微信保留独立历史，复用统一应用服务中的记忆、工具及调度。
        参数:
            message: 已获准且已持久化的输入。
            session_id: 恢复处理中请求时使用原请求的固定会话。
        返回:
            已选择正确会话的轻量 RuntimeClient。
        行为说明:
            1. 已有绑定精确恢复原 session_id，不使用最近会话推断。
            2. 新对话取得客户端初始会话后绑定；不在通道创建真实 Runtime。
        """
        session_id = session_id or self.store.binding(message)
        source = self.source_for(message)
        client = self.clients.get(source)
        if client is None:
            client = self.client_factory(source=source, session_id=session_id)
            self.clients[source] = client
        elif session_id is None:
            client.new_session()
        elif client.session_id != session_id:
            client.resume_session(session_id)
        if session_id is None:
            self.store.bind(message, client.session_id)
        return client

    def _answer(self, record: StoredMessage, client: RuntimeClient) -> PreparedReply:
        """处理本地命令，或以稳定请求 ID 提交统一应用服务。

        作用:
            把通道管理与模型输入分开，命令不作为自然语言任务重复执行。
        参数:
            record: 已标记为 processing 的输入。
            client: 当前输入所绑定的应用客户端。
        返回:
            完整答复和来源类别；模型答复附带 turn_id。
        行为说明:
            1. 媒体、空白和已支持的斜杠命令在本地处理，不调用模型。
            2. /retry 只复制已保存答案；/new 新建并保存新的对话绑定。
            3. /new 与普通文字使用已落盘请求 ID；模型内部循环由统一服务执行。
        """
        text = record.message.text
        if text is None:
            return PreparedReply(UNSUPPORTED_TEXT, "command")
        if not text.strip():
            return PreparedReply("请发送非空的文字消息。", "command")
        if text.strip().startswith("/"):
            command, _, argument = text.strip().partition(" ")
            argument = argument.strip()
            if command == "/help" and not argument:
                return PreparedReply(HELP_TEXT, "command")
            if command == "/session" and not argument:
                info = client.session_info
                return PreparedReply(f"当前会话：{info.session_id[:8]}\n标题：{info.title or '未命名'}", "command")
            if command == "/new":
                if len(argument) > 100:
                    return PreparedReply("会话标题请控制在 100 个字符以内。", "command")
                session = client.new_session(title=argument or None, request_id=record.backend_request_id)
                return PreparedReply(
                    f"已开启新会话：{session.session_id[:8]}。旧记录和全局记忆保留。",
                    "command",
                    binding_session_id=session.session_id,
                )
            if command == "/retry" and not argument:
                reply = self.store.latest_model_reply(record.message)
                return PreparedReply(reply or "当前会话还没有已生成的模型回答可以重发。", "command")
            return PreparedReply("命令或参数不支持。\n" + HELP_TEXT, "command")
        result = client.respond(text, request_id=record.backend_request_id)
        content = result.content or "本轮已结束，但模型没有返回可显示的文字。请检查会话记录。"
        return PreparedReply(content, "model", result.turn_id)

    @staticmethod
    def _operation(message: InboundMessage) -> str | None:
        """识别本条输入是否会产生应用后台操作。

        作用:
            只给模型请求与合法 /new 建立执行关联，纯展示命令留在通道处理。
        参数:
            message: 已持久化输入。
        返回:
            respond、new_session 或 None。
        行为说明:
            1. 仅解释命令，不调用模型或修改会话。
        """
        text = (message.text or "").strip()
        if not text:
            return None
        if not text.startswith("/"):
            return "respond"
        command, _, argument = text.partition(" ")
        return "new_session" if command == "/new" and len(argument.strip()) <= 100 else None

    def _save_reply(self, record: StoredMessage, reply: PreparedReply, *, error_code: str | None = None) -> None:
        """原子保存完整回复、分段及可选会话切换。

        作用:
            无论首次完成还是查回旧请求，都沿同一个投递准备入口落盘。
        参数:
            record: processing 输入记录。
            reply: 已生成且不会重新执行的完整结果。
            error_code: 可选受控错误类别。
        返回:
            None。
        行为说明:
            1. /new 的绑定和命令答复在同一事务写入，之后才允许处理后续消息。
        """
        self.store.prepare_reply(
            record.id,
            reply.text,
            split_reply(reply.text, self.reply_byte_limit),
            kind=reply.kind,
            turn_id=reply.turn_id,
            error_code=error_code,
            binding_session_id=reply.binding_session_id,
        )

    def _recover_request(self, record: StoredMessage) -> bool:
        """查询原应用请求并回收已完成结果，不重新提交任务。

        作用:
            连接器重启不代表后台执行中断，恢复必须以请求账本为准。
        参数:
            record: 带稳定后台请求关联的 processing 记录。
        返回:
            已保存终态回复时为 True；仍排队或运行时为 False。
        异常:
            ValueError: 待恢复记录不属于当前获准的通道、账号或本人。
        行为说明:
            1. 已完成 /new 恢复原目标会话；已完成模型回合只缓存原答案。
            2. 不存在、身份不符或无法确认的结果都转为保守通知，不重跑工具。
        """
        if (
            record.message.channel != self.adapter.channel
            or record.message.account_id != self.adapter.account_id
            or record.message.sender_id != self.adapter.owner_id
        ):
            raise ValueError("待恢复请求不属于当前获准通道，请检查本地通道记录")
        client = self._select_session(record.message, session_id=record.session_id)
        request = client.lookup_request(record.backend_request_id)
        valid = (
            isinstance(request, dict)
            and request.get("request_id") == record.backend_request_id
            and request.get("source") == self.source_for(record.message)
            and request.get("operation") == self._operation(record.message)
            and request.get("session_id") == record.session_id
        )
        if valid and request.get("status") in {"queued", "running"}:
            return False
        result = request.get("result") if valid else None
        reply = PreparedReply(INTERRUPTED_REPLY, "error")
        error_code = "request_interrupted"
        if valid and request.get("status") == "completed" and result is not None:
            if request["operation"] == "new_session" and getattr(result, "session_id", None):
                reply = PreparedReply(
                    f"已开启新会话：{result.session_id[:8]}。旧记录和全局记忆保留。",
                    "command",
                    binding_session_id=result.session_id,
                )
                error_code = None
            elif request["operation"] == "respond" and isinstance(getattr(result, "content", None), str):
                reply = PreparedReply(
                    result.content or "本轮已结束，但模型没有返回可显示的文字。请检查会话记录。",
                    "model",
                    getattr(result, "turn_id", None),
                )
                error_code = None
        self._save_reply(record, reply, error_code=error_code)
        self.report("已核对上次的微信请求状态；没有重新执行模型或工具。")
        return True

    def process_next(self) -> bool:
        """执行最早的一条待处理输入，并保存完整答案和稳定出站片段。

        作用:
            把执行与发送分开；答案保存后，发送失败无需再次调用模型或工具。
        返回:
            是否处理了一条输入 (bool)。
        异常:
            ValueError: 存储中出现不属于本通道或本人的待处理记录。
            捕获的原异常: 应用请求仍在排队或执行，结果无法确认时保留关联并重新抛出。
        行为说明:
            1. 执行前保存 processing，进程意外退出后该输入不会自动再执行。
            2. 请求异常先查账本；已完成则回收答案，结果未知保持 processing，不写假失败终态。
            3. 保存回复失败直接向外传播，不用错误通知覆盖已生成的答案。
        """
        self.start()
        if self.stop_event.is_set():
            return False
        pending = self.store.pending_requests(self.adapter.channel, self.adapter.account_id)
        if pending:
            return self._recover_request(pending[0])
        record = self.store.next_queued(self.adapter.channel, self.adapter.account_id)
        if record is None:
            return False
        message = record.message
        if (
            message.channel != self.adapter.channel
            or message.account_id != self.adapter.account_id
            or message.sender_id != self.adapter.owner_id
        ):
            raise ValueError("待处理输入不属于当前获准通道，请检查本地通道记录")
        client = self._select_session(message)
        identity = json.dumps([message.channel, message.account_id, message.conversation_id, message.message_id])
        request_id = f"channel:{uuid5(NAMESPACE_URL, identity).hex}" if self._operation(message) else None
        self.store.claim(record.id, client.session_id, request_id=request_id)
        record = self.store.get(record.id)
        error_code = None
        try:
            reply = self._answer(record, client)
        except KeyboardInterrupt:
            self.report("本轮已中断；没有自动重试，重启后将提示检查执行结果。")
            raise
        except Exception as error:
            if record.backend_request_id is not None:
                if self._recover_request(record):
                    return True
                self.report("微信原请求结果尚未确认，已保留请求关联；重启后将查询原结果，不重跑任务。")
                raise
            error_code = type(error).__name__
            reply = PreparedReply(FAILURE_TEXT, "error")
            self.report("本轮执行失败；详细执行记录留在 Kairos 会话中，未自动重试。")
        self._save_reply(record, reply, error_code=error_code)
        self.report(f"已处理一条微信输入，会话 {client.session_id[:8]}。")
        return True

    def flush_replies(self) -> int:
        """按顺序提交已保存的出站片段，不运行模型。

        作用:
            恢复尚未开始发送的答案，并为每段保存发送前后的状态。
        返回:
            本次被平台接口接受的片段数 (int)，不代表手机阅读回执。
        异常:
            TransportError: 发送失败或结果未知；保留答案后停止本次投递。
        行为说明:
            1. 已经开始但结果未知的发送不在 ready 列表中，不会自动重复。
            2. 发生异常时保留缓存，用户可用 /retry 明确要求再次发送答案。
        """
        self.start()
        sent = 0
        for record in self.store.ready_replies(self.adapter.channel, self.adapter.account_id):
            target = ReplyTarget(record.message.channel, record.message.account_id, record.message.conversation_id)
            checkpoint = self.store.checkpoint(target.channel, target.account_id) or {}
            for index, part in enumerate(record.parts):
                if self.stop_event.is_set():
                    return sent
                if part["status"] != "ready":
                    continue
                self.store.mark_sending(record.id, index)
                try:
                    receipt = self.adapter.send(target, part["text"], part["client_id"], checkpoint)
                except Exception as error:
                    self.store.mark_uncertain(record.id, index, "send_failed")
                    self.report("回复发送未能确认；已保存答案，没有重跑任务。可在微信发送 /retry 重发。")
                    raise ReplyDeliveryError(
                        "回复发送失败，结果未知",
                        uncertain=True,
                        authentication=isinstance(error, TransportError) and error.authentication,
                    ) from None
                self.store.mark_submitted(record.id, index, receipt.acknowledged)
                sent += 1
        return sent

    def _memory_events(self) -> None:
        """消费后台记忆状态，只在本机显示不含正文的结果。

        作用:
            防止长期运行时完成事件不断积累，同时不插入额外微信通知。
        返回:
            None。
        行为说明:
            1. 只展示受控的成功、冲突或失败类别，不输出主题、来源或错误原文。
        """
        for client in self.clients.values():
            for event in client.drain_memory_events():
                status = event.get("status")
                if status in {"failed", "partial", "conflict"}:
                    self.report("后台记忆处理有未完成或冲突项，请在本机检查记忆状态。")
                elif status == "completed":
                    self.report("后台记忆处理已完成。")

    def run(self, *, max_messages: int = 0, idle_timeout: float | None = None, receive: bool = True) -> int:
        """串行接收、执行并发送，直到达到验收上限或被用户停止。

        作用:
            在同一线程维护 Runtime 和 SQLite 的完整生命周期。
        参数:
            max_messages: 零表示持续运行，正数表示本次最多处理的输入数量。
            idle_timeout: 可选空闲秒数；长轮询返回后检查，不中断正在执行的回合。
            receive: False 时接收由独立线程负责，本循环只处理和发送已持久输入。
        返回:
            本次处理的输入数量 (int)，包含本地命令和错误通知。
        异常:
            ValueError: 消息上限为负或空闲时限不是正数。
        行为说明:
            1. 先发缓存答案，再处理已保存输入，最后从平台接收新消息。
            2. 仅可恢复的传输错误按有界退避重连，最多连续五次；不重跑任务。
            3. finally 仅分离本入口客户端，不关闭共享 Runtime；存储由调用方关闭。
        """
        if max_messages < 0 or (idle_timeout is not None and (not math.isfinite(idle_timeout) or idle_timeout <= 0)):
            raise ValueError("消息上限不能为负，空闲时限必须为正")
        processed, failures = 0, 0
        last_activity = time.monotonic()
        try:
            self.start()
            while not self.stop_event.is_set():
                self._memory_events()
                try:
                    self.flush_replies()
                    if max_messages and processed >= max_messages:
                        break
                    if self.process_next():
                        processed += 1
                        last_activity = time.monotonic()
                        continue
                    if self.store.pending_requests(self.adapter.channel, self.adapter.account_id):
                        last_activity = time.monotonic()
                    if idle_timeout is not None and time.monotonic() - last_activity >= idle_timeout:
                        break
                    count = self.poll_once() if receive else 0
                    failures = 0
                    if count:
                        last_activity = time.monotonic()
                    else:
                        self.stop_event.wait(0.2)
                except ReplyDeliveryError as error:
                    if error.authentication:
                        raise
                    # 未知投递已离开自动发送队列；继续收取 /retry 或其他新输入。
                    # 这里不会重试出站请求，也不会重新调用模型或工具。
                    continue
                except TransportError as error:
                    failures += 1
                    if error.authentication or not error.retryable or failures >= 5:
                        raise
                    self.report("微信连接暂时不可用，正在等待重连；不会重跑已执行的任务。")
                    self.stop_event.wait(min(2**failures, 30))
            return processed
        finally:
            self.close()

    def close(self) -> None:
        """分离当前通道持有的应用客户端。

        作用:
            停止当前入口的订阅，不关闭统一应用服务或真实业务 Runtime。
        返回:
            None。
        行为说明:
            1. 多次调用安全；不删除会话、通道记录或登录状态。
        """
        self._memory_events()
        clients, self.clients = self.clients, {}
        for client in clients.values():
            client.close()
