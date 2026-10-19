"""持久保存通道输入、会话绑定与发送状态，不执行模型或网络操作。"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .base import InboundMessage, ReceiveBatch

SCHEMA = """
CREATE TABLE IF NOT EXISTS channel_bindings (
    channel TEXT NOT NULL,
    account_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    PRIMARY KEY (channel, account_id, conversation_id)
);
CREATE TABLE IF NOT EXISTS channel_checkpoints (
    channel TEXT NOT NULL,
    account_id TEXT NOT NULL,
    checkpoint TEXT NOT NULL,
    PRIMARY KEY (channel, account_id)
);
CREATE TABLE IF NOT EXISTS channel_inbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT NOT NULL,
    account_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    sender_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    text TEXT,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'processing', 'ready', 'sending', 'sent', 'delivery_uncertain')),
    session_id TEXT,
    turn_id TEXT,
    response TEXT,
    kind TEXT,
    error_code TEXT,
    parts TEXT NOT NULL DEFAULT '[]',
    UNIQUE (channel, account_id, conversation_id, message_id)
);
CREATE INDEX IF NOT EXISTS channel_inbox_status ON channel_inbox(status, id);
CREATE INDEX IF NOT EXISTS channel_inbox_conversation
    ON channel_inbox(channel, account_id, conversation_id, session_id, id);
CREATE TABLE IF NOT EXISTS channel_request_links (
    inbox_id INTEGER PRIMARY KEY REFERENCES channel_inbox(id),
    request_id TEXT NOT NULL UNIQUE
);
"""

INTERRUPTED_REPLY = (
    "上次处理在完成前中断，结果可能不完整。我没有自动重新执行模型或工具；请先检查实际结果，再发送新的指令。"
)


@dataclass(frozen=True)
class StoredMessage:
    """表示一条持久输入及其处理、回复进度；各次读取互不共享可变数据。"""

    id: int
    message: InboundMessage
    status: str
    session_id: str | None
    turn_id: str | None
    response: str | None
    kind: str | None
    error_code: str | None
    parts: list[dict]
    backend_request_id: str | None = None


def _nonempty(value: str, label: str) -> str:
    """校验标识或状态字段为非空字符串。

    作用:
        在写数据库前拒绝无法关联或解释的参数。
    参数:
        value: 待校验的字段，不对实际值做转换。
        label: 可安全展示的字段名称。
    返回:
        原始字符串 (str)。
    异常:
        ValueError: 值不是非空字符串。
    行为说明:
        1. 错误只包含字段名，不包含消息、凭据或参数原文。
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 必须是非空字符串")
    return value


def _identity(message: InboundMessage) -> tuple[str, str, str]:
    """取得外部对话的完整身份。

    作用:
        为绑定与收件箱共用隔离边界，不混用同名平台会话。
    参数:
        message: 统一通道输入。
    返回:
        通道、账号、对话三元组 (tuple[str, str, str])。
    行为说明:
        1. 校验各个身份字段后保留原值。
    """
    return (
        _nonempty(message.channel, "channel"),
        _nonempty(message.account_id, "account_id"),
        _nonempty(message.conversation_id, "conversation_id"),
    )


def _scope(channel: str | None, account_id: str | None) -> tuple[str | None, str | None]:
    """校验队列与恢复操作的可选账号范围。

    作用:
        防止新登录账号读取或恢复旧账号积压；保留显式全局查询能力。
    参数:
        channel: 可选通道标识，须与 account_id 同时给出。
        account_id: 可选账号标识，须与 channel 同时给出。
    返回:
        已校验的通道和账号二元组；同时为 None 表示全局。
    异常:
        ValueError: 只提供通道或账号其中之一。
    行为说明:
        1. 不修改状态，实际查询继续使用参数绑定。
    """
    if (channel is None) != (account_id is None):
        raise ValueError("channel 与 account_id 必须同时提供或同时省略")
    if channel is None:
        return None, None
    return _nonempty(channel, "channel"), _nonempty(account_id, "account_id")


def _json(value: object) -> str:
    """把可持久数据编码为严格 JSON。

    作用:
        统一检查点与分段的序列化规则。
    参数:
        value: 可被标准 JSON 表达的数据。
    返回:
        Unicode JSON 字符串 (str)。
    行为说明:
        1. 禁止 NaN 等非标准数值，序列化失败时不生成替代内容。
    """
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _parts(texts: Sequence[str]) -> list[dict]:
    """生成一次完整回复的稳定发送分段。

    作用:
        在实际网络请求前固定文字和 client_id，重启后继续使用原标识。
    参数:
        texts: 按发送顺序排列的非空文字段，不接受单个字符串。
    返回:
        初始状态均为 ready 的分段字典列表 (list[dict])。
    异常:
        ValueError: 分段不是有效非空序列。
    行为说明:
        1. 校验全部文字后创建标识，不进行网络发送。
    """
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence) or not texts:
        raise ValueError("parts 必须是非空文字序列")
    for text in texts:
        _nonempty(text, "part.text")
    return [
        {"text": text, "client_id": f"kairos:{uuid4().hex}", "status": "ready", "acknowledged": None} for text in texts
    ]


class ChannelStore:
    """在当前线程串行管理独立通道 SQLite，不共享 Runtime 的业务连接。"""

    def __init__(self, path: str | Path):
        """以私有文件创建或打开通道数据库。

        作用:
            为协调层提供独立持久状态，不初始化 Runtime 或读取用户配置。
        参数:
            path: 数据库路径；测试必须传隔离临时路径。
        返回:
            无返回值。
        异常:
            ValueError: 数据库路径是符号链接。
        行为说明:
            1. 新目录使用 700，新数据库使用 600；已有目录和文件不更改权限。
            2. 初始化失败时关闭本次连接；实例只在创建连接的线程使用。
        """
        path = Path(path).expanduser()
        if path.is_symlink():
            raise ValueError("通道数据库不能是符号链接")
        missing = []
        parent = path.parent
        while not parent.exists():
            missing.append(parent)
            parent = parent.parent
        for directory in reversed(missing):
            directory.mkdir(mode=0o700, exist_ok=True)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.executescript(SCHEMA)
        except BaseException:
            self.connection.close()
            raise

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """为一次状态变化建立原子事务。

        作用:
            把输入、检查点或发送分段的相关修改一起保存。
        参数:
            无。
        返回:
            不提供绑定值的上下文管理器。
        异常:
            RuntimeError: 已存在尚未结束的事务。
            捕获的原异常: 事务执行或提交失败，回滚后重新抛出。
        行为说明:
            1. 不在事务中请求模型或网络。
        """
        if self.connection.in_transaction:
            raise RuntimeError("通道事务不能嵌套")
        self.connection.execute("BEGIN")
        try:
            yield
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def _stored(self, row: sqlite3.Row) -> StoredMessage:
        """把数据库行还原为独立记录。

        作用:
            向协调层隐藏 SQLite 行及 JSON 编码细节。
        参数:
            row: 收件箱查询返回的一行。
        返回:
            含统一输入和发送分段的 StoredMessage。
        行为说明:
            1. 每次解析新的分段列表，调用方修改不会直接改动存储。
            2. 应用请求 ID 从新增关联表读取，不改旧表结构或历史 payload。
        """
        link = self.connection.execute(
            "SELECT request_id FROM channel_request_links WHERE inbox_id = ?", (row["id"],)
        ).fetchone()
        return StoredMessage(
            id=row["id"],
            message=InboundMessage(
                channel=row["channel"],
                account_id=row["account_id"],
                conversation_id=row["conversation_id"],
                sender_id=row["sender_id"],
                message_id=row["message_id"],
                text=row["text"],
            ),
            status=row["status"],
            session_id=row["session_id"],
            turn_id=row["turn_id"],
            response=row["response"],
            kind=row["kind"],
            error_code=row["error_code"],
            parts=json.loads(row["parts"]),
            backend_request_id=link["request_id"] if link else None,
        )

    def checkpoint(self, channel: str, account: str) -> dict | None:
        """读取一个通道账号的已保存接收位置。

        作用:
            供适配器从上次原子保存的位置继续收消息。
        参数:
            channel: 通道标识。
            account: 机器人账号标识。
        返回:
            独立检查点字典；从未保存时为 None。
        行为说明:
            1. 不解释适配器内部字段，不更新进度。
        """
        row = self.connection.execute(
            "SELECT checkpoint FROM channel_checkpoints WHERE channel = ? AND account_id = ?",
            (_nonempty(channel, "channel"), _nonempty(account, "account")),
        ).fetchone()
        return json.loads(row["checkpoint"]) if row is not None else None

    def seed_checkpoint(self, channel: str, account: str, checkpoint: dict) -> None:
        """仅在尚无记录时保存初始检查点。

        作用:
            接收已有登录状态的迁移起点，不覆盖通道已经取得的进度。
        参数:
            channel: 通道标识。
            account: 账号标识。
            checkpoint: 适配器提供的不透明状态。
        返回:
            None。
        异常:
            ValueError: 检查点不是字典。
        行为说明:
            1. 同一账号已有检查点时保持原值。
        """
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint 必须是字典")
        values = (_nonempty(channel, "channel"), _nonempty(account, "account"), _json(checkpoint))
        with self._transaction():
            self.connection.execute("INSERT INTO channel_checkpoints VALUES (?, ?, ?) ON CONFLICT DO NOTHING", values)

    def accept_batch(self, channel: str, account: str, batch: ReceiveBatch) -> int:
        """原子保存一批输入与对应检查点。

        作用:
            在推进接收游标前持久保存消息，重复投递不创建新的处理任务。
        参数:
            channel: 本次适配器通道。
            account: 本次适配器账号。
            batch: 统一消息批次与适配器检查点。
        返回:
            实际新增的收件箱记录数 (int)。
        异常:
            ValueError: 检查点类型、消息类型、所属通道账号或文字类型不合法。
        行为说明:
            1. 按通道、账号、对话及消息 ID 去重，保留第一次收到的原文。
            2. 任一写入失败时，消息和检查点一起回滚。
        """
        channel = _nonempty(channel, "channel")
        account = _nonempty(account, "account")
        if not isinstance(batch.checkpoint, dict):
            raise ValueError("checkpoint 必须是字典")
        checkpoint = _json(batch.checkpoint)
        values = []
        for message in batch.messages:
            if not isinstance(message, InboundMessage):
                raise ValueError("batch 必须包含 InboundMessage")
            identity = _identity(message)
            if identity[:2] != (channel, account):
                raise ValueError("消息不属于本次通道账号")
            if message.text is not None and not isinstance(message.text, str):
                raise ValueError("消息 text 必须是字符串或 None")
            values.append(
                (
                    *identity,
                    _nonempty(message.sender_id, "sender_id"),
                    _nonempty(message.message_id, "message_id"),
                    message.text,
                )
            )
        inserted = 0
        with self._transaction():
            for value in values:
                cursor = self.connection.execute(
                    """INSERT INTO channel_inbox
                       (channel, account_id, conversation_id, sender_id, message_id, text)
                       VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING""",
                    value,
                )
                inserted += cursor.rowcount
            self.connection.execute(
                """INSERT INTO channel_checkpoints VALUES (?, ?, ?)
                   ON CONFLICT(channel, account_id) DO UPDATE SET checkpoint = excluded.checkpoint""",
                (channel, account, checkpoint),
            )
        return inserted

    def binding(self, message: InboundMessage) -> str | None:
        """读取输入所属外部对话的 Kairos 会话绑定。

        作用:
            让重启后的后续输入继续同一个内部 Session。
        参数:
            message: 用于定位通道、账号及外部对话的输入。
        返回:
            绑定的 session_id；尚未绑定时为 None。
        行为说明:
            1. 不按最近会话猜测，不创建 Runtime 会话。
        """
        row = self.connection.execute(
            "SELECT session_id FROM channel_bindings WHERE channel = ? AND account_id = ? AND conversation_id = ?",
            _identity(message),
        ).fetchone()
        return row["session_id"] if row is not None else None

    def bind(self, message: InboundMessage, session_id: str) -> None:
        """保存或替换一个外部对话的内部会话绑定。

        作用:
            连接首次创建会话及用户显式新建会话后的路由。
        参数:
            message: 外部对话身份。
            session_id: 已由 Runtime 创建的内部 Session ID。
        返回:
            None。
        行为说明:
            1. 仅更新当前三元身份，不修改其他聊天或旧输入记录。
        """
        values = (*_identity(message), _nonempty(session_id, "session_id"))
        with self._transaction():
            self.connection.execute(
                """INSERT INTO channel_bindings VALUES (?, ?, ?, ?)
                   ON CONFLICT(channel, account_id, conversation_id) DO UPDATE SET session_id = excluded.session_id""",
                values,
            )

    def next_queued(self, channel: str | None = None, account_id: str | None = None) -> StoredMessage | None:
        """取得最早等待处理的输入。

        作用:
            让主线程按持久收件顺序串行处理。
        参数:
            channel: 可选通道范围，与 account_id 同时给出。
            account_id: 可选账号范围；同时省略两个参数时查询所有账号。
        返回:
            最早 queued 记录；没有时为 None。
        行为说明:
            1. 本方法只读取；开始处理必须另行调用 claim。
            2. 指定范围时，其他账号更早的积压不会阻挡当前账号。
        """
        channel, account_id = _scope(channel, account_id)
        row = self.connection.execute(
            """SELECT * FROM channel_inbox WHERE status = 'queued'
               AND (? IS NULL OR (channel = ? AND account_id = ?)) ORDER BY id LIMIT 1""",
            (channel, channel, account_id),
        ).fetchone()
        return self._stored(row) if row is not None else None

    def get(self, record_id: int) -> StoredMessage:
        """读取指定收件箱记录的当前状态。

        作用:
            在每次状态变化后向协调层提供已持久结果。
        参数:
            record_id: 严格正整数本地记录 ID。
        返回:
            当前 StoredMessage。
        异常:
            ValueError: ID 不是严格正整数。
            KeyError: 指定记录不存在。
        行为说明:
            1. 不使用外部消息 ID 替代本地收件箱 ID。
        """
        if type(record_id) is not int or record_id <= 0:
            raise ValueError("record_id 必须是正整数")
        row = self.connection.execute("SELECT * FROM channel_inbox WHERE id = ?", (record_id,)).fetchone()
        if row is None:
            raise KeyError("通道消息不存在")
        return self._stored(row)

    def ready_replies(self, channel: str | None = None, account_id: str | None = None) -> list[StoredMessage]:
        """按接收顺序列出可以继续发送的完整回复。

        作用:
            为正常发送及重启恢复后的剩余分段提供队列。
        参数:
            channel: 可选通道范围，与 account_id 同时给出。
            account_id: 可选账号范围；同时省略两个参数时查询所有账号。
        返回:
            状态为 ready 的记录列表 (list[StoredMessage])。
        行为说明:
            1. 不返回 sending 或 delivery_uncertain，防止自动重发未知结果。
            2. 指定范围时，不返回其他账号等待发送的回复。
        """
        channel, account_id = _scope(channel, account_id)
        return [
            self._stored(row)
            for row in self.connection.execute(
                """SELECT * FROM channel_inbox WHERE status = 'ready'
                   AND (? IS NULL OR (channel = ? AND account_id = ?)) ORDER BY id""",
                (channel, channel, account_id),
            )
        ]

    def claim(self, record_id: int, session_id: str, *, request_id: str | None = None) -> None:
        """在提交应用请求前持久标记输入已经开始处理。

        作用:
            保证异常退出后这条输入不会再次自动执行模型或工具。
        参数:
            record_id: queued 记录 ID。
            session_id: 本次实际选择的内部会话。
            request_id: 会产生后台操作时使用的稳定请求 ID；纯本地命令可为空。
        返回:
            None。
        异常:
            RuntimeError: 输入已不在 queued 状态。
        行为说明:
            1. 同一事务保存会话、处理边界及请求关联，不调用应用服务。
        """
        session_id = _nonempty(session_id, "session_id")
        if request_id is not None:
            request_id = _nonempty(request_id, "request_id")
        with self._transaction():
            if self.get(record_id).status != "queued":
                raise RuntimeError("只有 queued 消息可以开始处理")
            self.connection.execute(
                "UPDATE channel_inbox SET status = 'processing', session_id = ? WHERE id = ?",
                (session_id, record_id),
            )
            if request_id is not None:
                self.connection.execute("INSERT INTO channel_request_links VALUES (?, ?)", (record_id, request_id))

    def pending_requests(self, channel: str, account_id: str) -> list[StoredMessage]:
        """读取需要与应用服务核对的处理中输入。

        作用:
            连接器重启后先恢复原请求，而不是重新执行或让后续 /new 越过它。
        参数:
            channel: 当前通道标识。
            account_id: 当前机器人账号。
        返回:
            按接收顺序排列的 processing 记录列表。
        行为说明:
            1. 只返回已关联请求 ID 的记录，旧记录仍由保守恢复处理。
        """
        channel, account_id = _scope(channel, account_id)
        return [
            self._stored(row)
            for row in self.connection.execute(
                """SELECT i.* FROM channel_inbox i JOIN channel_request_links r ON r.inbox_id = i.id
                   WHERE i.channel = ? AND i.account_id = ? AND i.status = 'processing' ORDER BY i.id""",
                (channel, account_id),
            )
        ]

    def prepare_reply(
        self,
        record_id: int,
        text: str,
        parts: Sequence[str],
        *,
        kind: str = "model",
        turn_id: str | None = None,
        error_code: str | None = None,
        binding_session_id: str | None = None,
    ) -> None:
        """在发送前保存完整回答及一次生成的分段标识。

        作用:
            把模型执行与网络发送分开，发送失败时可以查阅原回答而不重跑工具。
        参数:
            record_id: processing 输入记录 ID。
            text: 完整回答正文。
            parts: 按顺序发送的文字段。
            kind: 回答来源，默认 model；命令或错误使用独立类型。
            turn_id: 可选 Runtime 回合 ID。
            error_code: 可选受控错误码，不应包含原始异常或凭据。
            binding_session_id: /new 成功后的新绑定，与命令回复一起原子保存。
        返回:
            None。
        异常:
            RuntimeError: 输入不是 processing。
        行为说明:
            1. 全部字段和分段在一次事务中保存，之后不得再次生成 client_id。
        """
        _nonempty(text, "response")
        _nonempty(kind, "kind")
        if turn_id is not None:
            _nonempty(turn_id, "turn_id")
        if error_code is not None:
            _nonempty(error_code, "error_code")
        if binding_session_id is not None:
            _nonempty(binding_session_id, "binding_session_id")
        payload = _json(_parts(parts))
        with self._transaction():
            record = self.get(record_id)
            if record.status != "processing":
                raise RuntimeError("只有 processing 消息可以准备回复")
            if binding_session_id is not None:
                self.connection.execute(
                    """INSERT INTO channel_bindings VALUES (?, ?, ?, ?)
                       ON CONFLICT(channel, account_id, conversation_id)
                       DO UPDATE SET session_id = excluded.session_id""",
                    (*_identity(record.message), binding_session_id),
                )
            self.connection.execute(
                """UPDATE channel_inbox SET status = 'ready', response = ?, parts = ?, kind = ?,
                   turn_id = ?, error_code = ?, session_id = COALESCE(?, session_id) WHERE id = ?""",
                (text, payload, kind, turn_id, error_code, binding_session_id, record_id),
            )

    def _part(self, record_id: int, part_index: int, status: str, part_status: str) -> StoredMessage:
        """校验一次分段状态变化的前置条件。

        作用:
            避免越界、跳段、重发和对未知投递结果继续发送。
        参数:
            record_id: 收件箱记录 ID。
            part_index: 从零开始的分段索引。
            status: 要求的整条记录状态。
            part_status: 要求的目标分段状态。
        返回:
            包含目标分段的完整 StoredMessage。
        异常:
            ValueError: 索引不合法。
            RuntimeError: 记录、分段或先前分段状态不符合要求。
        行为说明:
            1. 只读取和校验，调用方负责在同一事务中写入。
        """
        record = self.get(record_id)
        if type(part_index) is not int or not 0 <= part_index < len(record.parts):
            raise ValueError("part_index 超出回复分段范围")
        if record.status != status or record.parts[part_index]["status"] != part_status:
            raise RuntimeError("回复或分段状态不允许此操作")
        if any(part["status"] != "submitted" for part in record.parts[:part_index]):
            raise RuntimeError("必须先提交前面的回复分段")
        return record

    def mark_sending(self, record_id: int, part_index: int) -> None:
        """在网络发送前持久标记一个分段正在发送。

        作用:
            将可能已产生副作用的发送与仍可安全发送的分段区分。
        参数:
            record_id: ready 记录 ID。
            part_index: 当前第一个未提交分段索引。
        返回:
            None。
        行为说明:
            1. 不创建新 client_id；异常退出后由 recover 停止自动发送。
        """
        with self._transaction():
            record = self._part(record_id, part_index, "ready", "ready")
            record.parts[part_index]["status"] = "sending"
            self.connection.execute(
                "UPDATE channel_inbox SET status = 'sending', parts = ? WHERE id = ?",
                (_json(record.parts), record_id),
            )

    def mark_submitted(self, record_id: int, part_index: int, acknowledged: bool) -> None:
        """保存一个分段被接口接受的结果及确认依据。

        作用:
            正常继续剩余分段，全部提交后结束整条回复。
        参数:
            record_id: sending 记录 ID。
            part_index: 正在发送的分段索引。
            acknowledged: 是否有明确业务成功码，不表示手机已读。
        返回:
            None。
        异常:
            ValueError: acknowledged 不是布尔值。
        行为说明:
            1. 分段统一标记 submitted 并保留确认依据，整条完成时标记 sent。
        """
        if type(acknowledged) is not bool:
            raise ValueError("acknowledged 必须是布尔值")
        with self._transaction():
            record = self._part(record_id, part_index, "sending", "sending")
            record.parts[part_index].update(status="submitted", acknowledged=acknowledged)
            status = "sent" if all(part["status"] == "submitted" for part in record.parts) else "ready"
            self.connection.execute(
                "UPDATE channel_inbox SET status = ?, parts = ? WHERE id = ?", (status, _json(record.parts), record_id)
            )

    def mark_uncertain(self, record_id: int, part_index: int, code: str) -> None:
        """保存发送异常并停止整条回复的自动发送。

        作用:
            在平台可能已经接收消息时避免重发或继续后续分段。
        参数:
            record_id: sending 记录 ID。
            part_index: 结果无法确认的分段。
            code: 可安全记录的错误码。
        返回:
            None。
        行为说明:
            1. 保留原文、client_id、已提交分段和未发送分段，便于用户显式决策。
        """
        code = _nonempty(code, "error_code")
        with self._transaction():
            record = self._part(record_id, part_index, "sending", "sending")
            record.parts[part_index]["status"] = "uncertain"
            self.connection.execute(
                "UPDATE channel_inbox SET status = 'delivery_uncertain', parts = ?, error_code = ? WHERE id = ?",
                (_json(record.parts), code, record_id),
            )

    def recover(self, channel: str | None = None, account_id: str | None = None) -> int:
        """修补上次异常退出的处理和发送边界。

        作用:
            重启后提供可追溯通知，不自动重新执行模型、工具或结果未知的发送。
        参数:
            channel: 可选通道范围，与 account_id 同时给出。
            account_id: 可选账号范围；同时省略两个参数时恢复所有账号。
        返回:
            本次修补的记录数量 (int)。
        行为说明:
            1. 无后台请求关联的 processing 转为固定中断通知；有关联的记录留待查询原请求。
            2. 遗留 sending 分段或记录转 delivery_uncertain；原 client_id 和回答保留。
            3. 已保存的 ready 回答继续可发；queued 输入仍可首次处理。
            4. 指定范围时，其他账号的处理和发送状态保持原样。
        """
        channel, account_id = _scope(channel, account_id)
        changed = 0
        with self._transaction():
            records = self.connection.execute(
                """SELECT * FROM channel_inbox WHERE status != 'sent'
                   AND (? IS NULL OR (channel = ? AND account_id = ?)) ORDER BY id""",
                (channel, channel, account_id),
            ).fetchall()
            for row in records:
                record = self._stored(row)
                if record.status == "sending" or any(part["status"] == "sending" for part in record.parts):
                    for part in record.parts:
                        if part["status"] == "sending":
                            part["status"] = "uncertain"
                    self.connection.execute(
                        """UPDATE channel_inbox SET status = 'delivery_uncertain', parts = ?,
                           error_code = 'delivery_interrupted' WHERE id = ?""",
                        (_json(record.parts), record.id),
                    )
                elif record.status == "processing" and record.backend_request_id is None:
                    self.connection.execute(
                        """UPDATE channel_inbox SET status = 'ready', response = ?, kind = 'error',
                           error_code = 'interrupted', parts = ? WHERE id = ?""",
                        (INTERRUPTED_REPLY, _json(_parts([INTERRUPTED_REPLY])), record.id),
                    )
                else:
                    continue
                changed += 1
        return changed

    def latest_model_reply(self, message: InboundMessage) -> str | None:
        """查询当前绑定会话里最近一次完整模型回答。

        作用:
            为用户显式请求重发提供缓存，只重发文字、不重新调用 Runtime。
        参数:
            message: 当前外部聊天身份。
        返回:
            已生成的完整模型回复；没有绑定或符合条件的记录时为 None。
        行为说明:
            1. 同时限定外部身份和当前 session_id，换会话后不误取旧会话答案。
            2. 不要求已成功发送；不返回命令、错误通知或处理中输入。
        """
        session_id = self.binding(message)
        if session_id is None:
            return None
        row = self.connection.execute(
            """SELECT response FROM channel_inbox WHERE channel = ? AND account_id = ? AND conversation_id = ?
               AND session_id = ? AND kind = 'model' AND response IS NOT NULL ORDER BY id DESC LIMIT 1""",
            (*_identity(message), session_id),
        ).fetchone()
        return row["response"] if row is not None else None

    def close(self) -> None:
        """关闭当前线程的通道数据库连接。

        作用:
            在协调层退出时释放资源，保留所有持久记录。
        参数:
            无。
        返回:
            None。
        行为说明:
            1. 不等待网络或模型任务；本存储没有后台线程。
        """
        self.connection.close()
