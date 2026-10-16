"""通用会话及其回合、步骤、消息和工具调用的 SQLite 存储。

只在完整记录产生时做短事务，不在模型请求或工具执行期间持有事务。
连接由 Runtime 提供和关闭；这里不改变调用方的 row_factory。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

SESSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    title_is_custom INTEGER NOT NULL DEFAULT 0,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    thinking_enabled INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS session_turns (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    user_message_id TEXT NOT NULL REFERENCES session_messages(message_id)
        DEFERRABLE INITIALLY DEFERRED,
    status TEXT NOT NULL,
    end_reason TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS session_steps (
    step_id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL REFERENCES session_turns(turn_id),
    step_index INTEGER NOT NULL CHECK (step_index >= 1),
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    usage TEXT,
    finish_reason TEXT,
    duration_ms REAL,
    error TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    UNIQUE (turn_id, step_index)
);
CREATE TABLE IF NOT EXISTS session_messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    seq INTEGER NOT NULL CHECK (seq >= 0),
    turn_id TEXT REFERENCES session_turns(turn_id),
    step_id TEXT REFERENCES session_steps(step_id),
    role TEXT NOT NULL,
    payload TEXT NOT NULL,
    tool_call_id TEXT,
    in_context INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    UNIQUE (session_id, seq),
    FOREIGN KEY (step_id, tool_call_id)
        REFERENCES session_tool_calls(step_id, tool_call_id)
        DEFERRABLE INITIALLY DEFERRED
);
CREATE TABLE IF NOT EXISTS session_tool_calls (
    step_id TEXT NOT NULL REFERENCES session_steps(step_id),
    tool_call_id TEXT NOT NULL,
    assistant_message_id TEXT NOT NULL REFERENCES session_messages(message_id),
    tool_name TEXT NOT NULL,
    arguments TEXT NOT NULL,
    status TEXT NOT NULL,
    result_message_id TEXT REFERENCES session_messages(message_id),
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    PRIMARY KEY (step_id, tool_call_id)
);
CREATE INDEX IF NOT EXISTS idx_session_turns_session ON session_turns(session_id);
CREATE INDEX IF NOT EXISTS idx_session_messages_turn ON session_messages(turn_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_session_assistant_step
    ON session_messages(step_id) WHERE role = 'assistant';
CREATE UNIQUE INDEX IF NOT EXISTS idx_session_tool_result
    ON session_messages(step_id, tool_call_id) WHERE role = 'tool';
CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);
"""


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    title: str
    provider: str
    model: str
    thinking_enabled: bool
    created_at: str
    updated_at: str
    message_count: int


@dataclass(frozen=True)
class MessageRecord:
    message_id: str
    session_id: str
    seq: int
    turn_id: str | None
    step_id: str | None
    role: str
    payload: dict
    in_context: bool


@dataclass(frozen=True)
class TurnRecord:
    turn_id: str
    session_id: str
    user_message_id: str
    status: str
    end_reason: str | None


@dataclass(frozen=True)
class StepRecord:
    step_id: str
    turn_id: str
    step_index: int
    model: str
    status: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _id() -> str:
    return str(uuid4())


def _json(value: Any) -> str:
    """编码完整数据为保留中文的标准 JSON 字符串。

    作用:
        为消息、工具参数和运行统计提供统一编码，供调用方写入数据库。
        只转换传入值，不写库，也不修改原对象。

    参数:
        value: 可 JSON 序列化的值，也可以是字段可序列化的数据类实例。

    返回:
        编码后的 JSON 文本(str)。None 编码为字符串 "null"。

    行为说明:
        1. 将数据类实例转换为字典，其余值直接交给编码器。
        2. 保留中文字符，并拒绝 NaN、Infinity 等非标准 JSON 数值。
    """
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}不能为空")
    return value.strip()


def _payload(payload: dict, role: str) -> str:
    """校验完整消息的角色并编码消息内容。

    作用:
        在消息写入前统一校验角色，输出供数据库保存的完整 JSON 文本。
        保留图片和模型协议字段，不修改消息，也不访问数据库。

    参数:
        payload: 可 JSON 序列化的消息字典，如 {"role": "user", "content": "你好"}。
        role: 期望的角色值，必须与 payload 中的 role 完全一致。

    返回:
        完整消息的 JSON 文本(str)。

    异常:
        ValueError: payload 不是字典，或其 role 与要求的角色不一致。

    行为说明:
        1. 检查消息是字典且角色符合调用方要求，不额外校验各协议字段的结构。
        2. 编码整个消息对象，编码失败时尚未产生数据库写入。
    """
    if not isinstance(payload, dict) or payload.get("role") != role:
        raise ValueError(f"消息必须是 role={role} 的完整消息对象")
    return _json(payload)


def _message(row: dict) -> MessageRecord:
    """还原数据库消息行，生成消息记录。

    作用:
        连接存储查询与 Runtime 或历史展示，恢复内部归属信息及完整模型消息。
        只转换查询结果，不读写数据库。

    参数:
        row: 包含 message_id、session_id、seq、turn_id、step_id、role、payload、
            in_context 的数据库行字典；payload 应为有效的 JSON 文本。

    返回:
        包含内部 ID、归属信息和解码后 payload 的消息记录(MessageRecord)。

    行为说明:
        1. 直接保留消息 ID、会话及执行过程的归属字段。
        2. 解码 payload，并将 in_context 转为布尔值；不额外校验消息内容结构。
    """
    return MessageRecord(
        message_id=row["message_id"],
        session_id=row["session_id"],
        seq=row["seq"],
        turn_id=row["turn_id"],
        step_id=row["step_id"],
        role=row["role"],
        payload=json.loads(row["payload"]),
        in_context=bool(row["in_context"]),
    )


def _automatic_title(payload: dict, hint: str | None) -> str:
    """提取标题提示或首条输入中的简短会话标题。

    作用:
        在首轮输入到来时为未手动命名的会话提供标题，由调用方负责保存。
        本函数只整理文本，不调用模型，也不写库。

    参数:
        payload: 用户消息字典；content 可以是文本或内容块列表。
            内容块示例: {"type": "text", "text": "讨论方案"}。
        hint: 可选标题提示；非空时优先于 payload.content。

    返回:
        第一行文本整理后的标题(str)，超过 40 个字符时截断并追加省略号。
        无可用文本时返回 "图片对话"。

    行为说明:
        1. 优先选择非空 hint，否则读取 content；列表只提取 text 类型内容块。
        2. 取首行、合并空白并限制长度，空文本使用默认标题。
    """
    content = hint or payload.get("content") or ""
    if isinstance(content, list):
        content = " ".join(block.get("text", "") for block in content if block.get("type") == "text")
    lines = str(content).strip().splitlines()
    title = " ".join(lines[0].split()) if lines else "图片对话"
    return title[:40] + ("…" if len(title) > 40 else "")


class SessionStore:
    """持久化通用会话对象；所有方法都在返回前结束自身的写事务。"""

    def __init__(self, connection: sqlite3.Connection) -> None:
        """初始化已有数据库连接上的会话存储结构。

        作用:
            在 Runtime 创建存储时启用外键并准备会话相关表和索引。
            保存连接引用，连接的创建和关闭仍由调用方负责。

        参数:
            connection: 可用的 SQLite 连接，必须没有尚未结束的事务。

        返回:
            无返回值(None)；初始化后的实例通过该连接执行后续存取。

        异常:
            RuntimeError: 连接已经处于事务中。
            捕获的原异常（显式重新抛出）: 建表脚本执行失败，执行回滚后重新抛出。

        行为说明:
            1. 保存连接引用并拒绝已有事务，避免影响调用方未完成的写入。
            2. 启用外键，在显式事务中创建缺失的表和索引，建表阶段失败时回滚。
            3. 保留连接供后续使用，不改变其 row_factory，也不负责关闭连接。
        """
        self.connection = connection
        if connection.in_transaction:
            raise RuntimeError("初始化会话存储前请先结束已有数据库事务")
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            connection.executescript("BEGIN;\n" + SESSION_SCHEMA + "\nCOMMIT;")
        except BaseException:
            connection.rollback()
            raise

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """建立一次会话写入的提交或回滚事务边界。

        作用:
            包裹调用方的一组短数据库操作，避免同一逻辑写入只完成一部分。
            使用实例持有的连接，不在模型请求或工具执行期间持有事务。

        参数:
            无显式参数；读取并操作 self.connection，要求连接当前没有活动事务。

        返回:
            用于 with 语句的上下文管理器；进入时不绑定业务值，生成器产出 None
            (Iterator[None])。

        异常:
            RuntimeError: 连接已经处于事务中，拒绝嵌套。
            捕获的原异常（显式重新抛出）: 事务体或提交阶段失败，执行回滚后重新抛出。

        行为说明:
            1. 检查没有已有事务，然后执行 BEGIN。
            2. 执行 with 块并在正常退出时提交；事务体或提交失败时尝试回滚。
            3. 事务只覆盖 with 块及提交过程，不涵盖退出后调用方的查询或其他操作。
        """
        if self.connection.in_transaction:
            raise RuntimeError("会话操作不能嵌套在其他数据库事务中")
        self.connection.execute("BEGIN")
        try:
            yield
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def _rows(self, sql: str, parameters: tuple = ()) -> list[dict]:
        """执行带参数的 SQL 并将结果行转换为字典列表。

        作用:
            为会话存储查询提供统一行结构，供内部读取和记录装配使用。
            使用独立游标，不修改共享连接的 row_factory，也不自行提交事务。

        参数:
            sql: 待执行的 SQL，调用方通常传入查询语句。
            parameters: 按 SQL 占位符顺序提供的参数元组，默认空元组。

        返回:
            查询结果的字典列表(list[dict])；没有结果行时返回空列表。

        行为说明:
            1. 创建游标，只为该游标设置 sqlite3.Row。
            2. 执行 SQL、读取全部结果并转换为普通字典，结束时关闭游标。
        """
        cursor = self.connection.cursor()
        cursor.row_factory = sqlite3.Row
        try:
            return [dict(row) for row in cursor.execute(sql, parameters).fetchall()]
        finally:
            cursor.close()

    def _one(self, sql: str, parameters: tuple, label: str) -> dict:
        rows = self._rows(sql, parameters)
        if not rows:
            raise ValueError(f"找不到{label}")
        return rows[0]

    def _session_info(self, session_id: str) -> SessionInfo:
        """读取指定会话的概要和有效消息数量。

        作用:
            为会话列表、恢复和写入后的返回值装配元信息，不修改数据库。

        参数:
            session_id: 已存在会话的完整 ID，不在此处解析 ID 前缀。

        返回:
            会话概要(SessionInfo)，含模型配置、时间及有效的非系统消息数量。

        行为说明:
            1. 查询会话并统计 in_context=1 且角色不是 system 的消息。
            2. 将思考开关转换为布尔值并装配概要，不加载消息正文。
        """
        row = self._one(
            """SELECT s.*, (SELECT COUNT(*) FROM session_messages m
                WHERE m.session_id = s.session_id AND m.role != 'system' AND m.in_context = 1) AS message_count
                FROM sessions s WHERE s.session_id = ?""",
            (session_id,),
            "会话",
        )
        return SessionInfo(
            session_id=row["session_id"],
            title=row["title"],
            provider=row["provider"],
            model=row["model"],
            thinking_enabled=bool(row["thinking_enabled"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            message_count=row["message_count"],
        )

    def _touch(self, session_id: str) -> None:
        """更新指定会话的最后活动时间。

        作用:
            在其他会话写入中同步刷新列表排序依据，由调用方负责事务边界。
            只执行更新时间的 SQL，不单独提交。

        参数:
            session_id: 要更新的完整会话 ID；本函数不检查该会话是否存在。

        返回:
            无返回值(None)；ID 不匹配任何会话时也不返回更新行数。

        行为说明:
            1. 用当前 UTC 时间更新目标行；通常在调用方已开启的事务内执行。
            2. 不检查或提交事务，也不将未找到目标行作为错误。
        """
        self.connection.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (_now(), session_id))

    def _insert_message(
        self,
        session_id: str,
        role: str,
        encoded_payload: str,
        *,
        message_id: str | None = None,
        turn_id: str | None = None,
        step_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> str:
        """追加消息到当前事务并建立归属关系。

        作用:
            为系统、用户、助手和工具结果消息提供共用写入入口，同时刷新会话活动时间。
            调用方负责校验消息、保证各归属一致，并与相关记录一并提交。

        参数:
            session_id: 消息所属会话的完整 ID。
            role: 消息角色；应与 encoded_payload 中的角色一致，本函数不重复校验。
            encoded_payload: 已校验并编码的完整消息 JSON 文本。
            message_id: 可选内部消息 ID；未提供或为空时生成新 ID。
            turn_id: 可选所属回合 ID，系统消息可以为空。
            step_id: 可选所属步骤 ID，用户消息和系统消息可以为空。
            tool_call_id: 可选模型工具调用 ID；工具结果用它与 step_id 共同关联原调用。

        返回:
            新写入消息的内部 ID(str)。

        行为说明:
            1. 确定消息 ID，并取该会话最大 seq 加一作为新序号，首条为 0。
            2. 保存消息及归属字段，再更新时间；本函数不自行开启或提交事务。
            3. 归属 ID 的存在性由外键约束保障，延迟外键在调用方提交时检查；
               跨层级归属一致性由调用方保证。
        """
        message_id = message_id or _id()
        row = self._rows(
            "SELECT COALESCE(MAX(seq), -1) + 1 AS seq FROM session_messages WHERE session_id = ?",
            (session_id,),
        )[0]
        self.connection.execute(
            """INSERT INTO session_messages
                (message_id, session_id, seq, turn_id, step_id, role, payload, tool_call_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (message_id, session_id, row["seq"], turn_id, step_id, role, encoded_payload, tool_call_id, _now()),
        )
        self._touch(session_id)
        return message_id

    def _get_message(self, message_id: str) -> MessageRecord:
        return _message(self._one("SELECT * FROM session_messages WHERE message_id = ?", (message_id,), "消息"))

    def _step(self, step_id: str) -> dict:
        """读取步骤记录及其所属会话 ID。

        作用:
            为助手消息、工具结果和状态写入提供步骤的回合与会话归属。
            只查询数据库，不改变步骤状态。

        参数:
            step_id: 已存在步骤的完整 ID。

        返回:
            包含步骤全部字段及 session_id 的字典(dict)。

        行为说明:
            1. 通过 turn_id 将步骤与回合关联，读取回合中的 session_id。
            2. 返回查询结果；状态是否合法由调用方法另行判断。
        """
        return self._one(
            """SELECT s.*, t.session_id FROM session_steps s JOIN session_turns t ON t.turn_id = s.turn_id
                WHERE s.step_id = ?""",
            (step_id,),
            "步骤",
        )

    def create_session(
        self,
        system_prompt: str,
        *,
        provider: str,
        model: str,
        thinking_enabled: bool = False,
        title: str | None = None,
    ) -> SessionInfo:
        """创建会话及其第一条系统消息。

        作用:
            在新建会话时保存模型配置和系统提示词，返回供 Runtime 选择的会话概要。
            会话 ID 和系统消息 ID 均在本地生成。

        参数:
            system_prompt: 系统提示词，作为 system 消息的 content 保存；可为空字符串。
            provider: 模型协议或提供方标识，原样保存，不在此处验证可用性。
            model: 模型名称，原样保存，不在此处验证模型是否可调用。
            thinking_enabled: 思考开关，默认 False，转换为整数保存。
            title: 可选手动标题；提供时必须是去除首尾空白后非空的字符串。

        返回:
            新创建会话的概要(SessionInfo)。

        行为说明:
            1. 准备标题、会话 ID 和系统消息 JSON，尚不写库。
            2. 在同一事务中插入会话和系统消息；默认标题为 "新会话"，首轮输入时再生成标题。
            3. 提交后读取并返回概要；此时查询失败不会撤销已提交的创建操作。
        """
        title_value = "新会话" if title is None else _nonempty(title, "会话标题")
        session_id, now = _id(), _now()
        payload = _payload({"role": "system", "content": system_prompt}, "system")
        with self._transaction():
            self.connection.execute(
                """INSERT INTO sessions
                    (session_id, title, title_is_custom, provider, model, thinking_enabled, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, title_value, int(title is not None), provider, model, int(thinking_enabled), now, now),
            )
            self._insert_message(session_id, "system", payload)
        return self._session_info(session_id)

    def get_session(self, reference: str) -> SessionInfo:
        """按完整 ID 或唯一前缀查找；前缀按字面匹配，不解释 SQL 通配符。"""
        reference = _nonempty(reference, "会话 ID")
        exact = self._rows("SELECT session_id FROM sessions WHERE session_id = ?", (reference,))
        if exact:
            return self._session_info(reference)
        matches = self._rows(
            "SELECT session_id FROM sessions WHERE substr(session_id, 1, length(?)) = ? LIMIT 2",
            (reference, reference),
        )
        if not matches:
            raise ValueError("找不到对应会话")
        if len(matches) > 1:
            raise ValueError("会话 ID 前缀匹配多个会话，请输入更长的 ID")
        return self._session_info(matches[0]["session_id"])

    def list_sessions(self, limit: int = 50) -> list[SessionInfo]:
        """列出按最后活动时间倒序排列的可见会话。

        作用:
            为会话列表和继续最近会话提供候选项，隐藏未命名且尚无回合的空会话。
            只查询元信息，不恢复历史或修改活动时间。

        参数:
            limit: 最多返回的会话数量，必须通过 int 类型检查且大于或等于 1，默认 50。

        返回:
            会话概要列表(list[SessionInfo])；没有符合条件的会话时返回空列表。

        异常:
            ValueError: limit 不是 int 或小于 1。

        行为说明:
            1. 校验数量，筛选手动命名或已创建过回合的会话。
            2. 按 updated_at、session_id 倒序取前 limit 项，再读取各自概要。
        """
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("会话列表数量必须大于或等于 1")
        rows = self._rows(
            """SELECT s.session_id FROM sessions s WHERE s.title_is_custom = 1 OR EXISTS
                (SELECT 1 FROM session_turns t WHERE t.session_id = s.session_id)
                ORDER BY s.updated_at DESC, s.session_id DESC LIMIT ?""",
            (limit,),
        )
        return [self._session_info(row["session_id"]) for row in rows]

    def latest_session(self) -> SessionInfo:
        """读取最近活动的可见会话。

        作用:
            为继续最近会话提供目标，沿用会话列表的可见性和排序规则。
            仅返回概要，不加载消息或修复中断状态。

        参数:
            无显式参数；通过 self 读取当前数据库中的会话。

        返回:
            最近活动会话的概要(SessionInfo)；无可见会话时抛异常，不返回 None。

        异常:
            ValueError: 会话列表为空，没有可以继续的历史会话。

        行为说明:
            1. 用 limit=1 查询会话列表。
            2. 返回唯一候选；列表为空时说明没有可以继续的历史会话。
        """
        sessions = self.list_sessions(limit=1)
        if not sessions:
            raise ValueError("还没有可以继续的历史会话")
        return sessions[0]

    def rename_session(self, session_id: str, title: str) -> SessionInfo:
        """保存会话的手动标题并刷新活动时间。

        作用:
            接收重命名操作，持久化标题供列表及后续恢复使用。
            手动标题标记会阻止首轮输入覆盖标题。

        参数:
            session_id: 已存在会话的完整 ID。
            title: 去除首尾空白后必须非空的标题字符串。

        返回:
            更新后的会话概要(SessionInfo)。

        行为说明:
            1. 检查会话存在并整理标题，此阶段没有写入。
            2. 在事务内保存标题、手动命名标记及活动时间。
            3. 提交后重新读取概要；读取失败不会撤销已提交的标题修改。
        """
        self._session_info(session_id)
        title = _nonempty(title, "会话标题")
        with self._transaction():
            self.connection.execute(
                "UPDATE sessions SET title = ?, title_is_custom = 1, updated_at = ? WHERE session_id = ?",
                (title, _now(), session_id),
            )
        return self._session_info(session_id)

    def update_thinking(self, session_id: str, enabled: bool) -> SessionInfo:
        """保存会话的思考开关并刷新活动时间。

        作用:
            将 Runtime 选择的思考开关持久化，供下次恢复会话时使用。
            本函数不修改模型客户端，也不发起模型请求。

        参数:
            session_id: 已存在会话的完整 ID。
            enabled: 要保存的布尔开关；按 int(enabled) 存储，不额外检查布尔类型。

        返回:
            更新后的会话概要(SessionInfo)。

        行为说明:
            1. 确认会话存在，再在事务内保存开关和活动时间。
            2. 提交后重新读取概要；该查询失败不会撤销已提交的开关修改。
        """
        self._session_info(session_id)
        with self._transaction():
            self.connection.execute(
                "UPDATE sessions SET thinking_enabled = ?, updated_at = ? WHERE session_id = ?",
                (int(enabled), _now(), session_id),
            )
        return self._session_info(session_id)

    def touch_session(self, session_id: str) -> SessionInfo:
        """更新并独立提交指定会话的活动时间。

        作用:
            在恢复或切换会话后刷新排序依据，使其按最近活动时间出现在列表中。
            供需要独立事务的调用方使用，区别于内部 _touch。

        参数:
            session_id: 已存在会话的完整 ID。

        返回:
            活动时间更新后的会话概要(SessionInfo)。

        行为说明:
            1. 检查会话存在，在独立事务内调用 _touch 并提交。
            2. 提交后读取概要；读取失败不会回滚已提交的时间更新。
        """
        self._session_info(session_id)
        with self._transaction():
            self._touch(session_id)
        return self._session_info(session_id)

    def load_messages(self, session_id: str, *, include_inactive: bool = False) -> list[MessageRecord]:
        """加载按会话内顺序排列的完整消息记录。

        作用:
            为 Runtime 恢复模型上下文或为历史展示提供消息，保留图片和协议字段。
            只读取数据库，不修复中断回合，也不改变消息状态。

        参数:
            session_id: 已存在会话的完整 ID。
            include_inactive: 是否同时返回 in_context=0 的消息，默认 False。
                设为 True 时可查看被撤回的用户输入。

        返回:
            按 seq 升序排列的消息记录列表(list[MessageRecord])；无匹配消息时返回空列表。

        行为说明:
            1. 确认会话存在，按选项决定是否只读取 in_context=1 的消息。
            2. 按 seq 排序读取，逐条解码为 MessageRecord，不对 payload 做裁剪。
        """
        self._session_info(session_id)
        condition = "" if include_inactive else " AND in_context = 1"
        return [
            _message(row)
            for row in self._rows(
                "SELECT * FROM session_messages WHERE session_id = ?" + condition + " ORDER BY seq",
                (session_id,),
            )
        ]

    def begin_turn(self, session_id: str, payload: dict, *, title_hint: str | None = None) -> TurnRecord:
        """创建回合及其发起用户消息并建立双向关联。

        作用:
            在 Runtime 接受新输入后记录本轮执行起点，供后续 Step 和消息引用。
            同时保存输入，首次输入还可能更新默认会话标题。

        参数:
            session_id: 已存在且没有 running 回合的会话完整 ID。
            payload: 可 JSON 序列化、role="user" 的完整消息字典。
                例如 {"role": "user", "content": "你好"}；图片内容也原样保存。
            title_hint: 可选标题提示，只在会话尚无回合且未手动命名时使用。

        返回:
            状态为 running 的新回合(TurnRecord)，含本地生成的 turn_id、user_message_id。

        异常:
            ValueError: 当前会话已有 status="running" 的回合。

        行为说明:
            1. 校验并编码输入，生成回合 ID 和消息 ID，尚不写库。
            2. 在同一事务内检查会话和已有回合，插入回合与用户消息；用户消息没有 step_id。
            3. 必要时生成首轮标题后一起提交，保证回合与发起消息的关联同时成立。
        """
        encoded = _payload(payload, "user")
        turn_id, message_id = _id(), _id()
        with self._transaction():
            session = self._one("SELECT * FROM sessions WHERE session_id = ?", (session_id,), "会话")
            previous = self._rows("SELECT status FROM session_turns WHERE session_id = ?", (session_id,))
            if any(row["status"] == "running" for row in previous):
                raise ValueError("当前会话还有未结束的回合，请先完成或恢复该回合")
            self.connection.execute(
                """INSERT INTO session_turns (turn_id, session_id, user_message_id, status, created_at)
                    VALUES (?, ?, ?, 'running', ?)""",
                (turn_id, session_id, message_id, _now()),
            )
            self._insert_message(session_id, "user", encoded, message_id=message_id, turn_id=turn_id)
            if not previous and not session["title_is_custom"]:
                self.connection.execute(
                    "UPDATE sessions SET title = ? WHERE session_id = ?",
                    (_automatic_title(payload, title_hint), session_id),
                )
        return TurnRecord(turn_id, session_id, message_id, "running", None)

    def start_step(self, turn_id: str, step_index: int, model: str) -> StepRecord:
        """创建运行中回合的一个执行步骤。

        作用:
            在 Agent 循环开始一次模型交互时保存步骤，供助手消息和工具调用归属。
            写入 running 状态并刷新所属会话的活动时间。

        参数:
            turn_id: 已存在且 status="running" 的回合 ID。
            step_index: 通过 int 类型检查且大于或等于 1 的回合内序号，不能与已有步骤重复。
                调用方负责递增，本函数不检查序号是否连续。
            model: 本步骤使用的模型名称，原样保存，不检查模型可用性。

        返回:
            状态为 running 的新步骤(StepRecord)，step_id 由本地独立生成。

        异常:
            ValueError: 满足以下任一情况时抛出:
                - step_index 不是 int 或小于 1；
                - 所属回合的 status 不是 "running"。

        行为说明:
            1. 校验序号并生成步骤 ID，尚不写库。
            2. 在事务内检查回合状态、插入步骤并更新会话活动时间，成功后一起提交。
        """
        if not isinstance(step_index, int) or step_index < 1:
            raise ValueError("步骤序号必须从 1 开始")
        step_id = _id()
        with self._transaction():
            turn = self._one("SELECT * FROM session_turns WHERE turn_id = ?", (turn_id,), "回合")
            if turn["status"] != "running":
                raise ValueError("不能在已结束的回合中开始步骤")
            self.connection.execute(
                """INSERT INTO session_steps (step_id, turn_id, step_index, model, status, created_at)
                    VALUES (?, ?, ?, ?, 'running', ?)""",
                (step_id, turn_id, step_index, model, _now()),
            )
            self._touch(turn["session_id"])
        return StepRecord(step_id, turn_id, step_index, model, "running")

    def append_assistant(self, step_id: str, payload: dict) -> MessageRecord:
        """追加助手消息到运行中的步骤并登记其中的工具调用。

        作用:
            在模型回复后保存完整消息，为后续工具执行建立 pending 调用记录。
            本地生成消息 ID，沿用模型返回的调用 ID，并用 (step_id, tool_call_id) 定位调用。

        参数:
            step_id: 已存在、status="running" 且尚未保存助手消息的步骤 ID。
            payload: 可 JSON 序列化、role="assistant" 的完整消息字典；tool_calls 可选。
                每个调用应为字典，例如 {"id": "call_1", "function": {
                "name": "search", "arguments": "{}"}}。
                调用 ID 必须是非空字符串且在本步骤内唯一，工具名称必须是非空字符串。

        返回:
            新保存的助手消息记录(MessageRecord)，包含会话、回合及步骤归属。

        异常:
            ValueError: 满足以下任一情况时抛出:
                - 工具调用 ID 不是字符串，或去除首尾空白后为空；
                - 同一条助手消息中的工具调用 ID 重复；
                - 所属步骤的 status 不是 "running"。

        行为说明:
            1. 编码完整消息并校验、整理调用，此阶段不写库。参数文本即使不是合法 JSON 也保留；
               非字符串参数只在 session_tool_calls.arguments 中转换为 JSON，不修改原 payload。
            2. 在一个事务内检查步骤状态、插入消息及整批 pending 调用，失败时回滚该批写入。
            3. 提交后读取消息记录；该读取失败不会撤销已提交的消息和调用。
        """
        encoded = _payload(payload, "assistant")
        calls = payload.get("tool_calls") or []
        ids: set[str] = set()
        normalized = []
        for call in calls:
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id.strip():
                raise ValueError("工具调用 ID 不能为空")
            if call_id in ids:
                raise ValueError("同一步骤中的工具调用 ID 不能重复")
            ids.add(call_id)
            function = call.get("function") or {}
            name = function.get("name")
            _nonempty(name, "工具名称")
            arguments = function.get("arguments", "{}")
            # 保存原始参数文本，包括尚未通过工具解析的无效 JSON，供审计使用。
            normalized.append((call_id, name, arguments if isinstance(arguments, str) else _json(arguments)))
        with self._transaction():
            step = self._step(step_id)
            if step["status"] != "running":
                raise ValueError("不能向已结束的步骤添加助手消息")
            message_id = self._insert_message(
                step["session_id"],
                "assistant",
                encoded,
                turn_id=step["turn_id"],
                step_id=step_id,
            )
            self.connection.executemany(
                """INSERT INTO session_tool_calls
                    (step_id, tool_call_id, assistant_message_id, tool_name, arguments, status, created_at)
                    VALUES (?, ?, ?, ?, ?, 'pending', ?)""",
                [(step_id, call_id, message_id, name, arguments, _now()) for call_id, name, arguments in normalized],
            )
        return self._get_message(message_id)

    def start_tool(self, step_id: str, tool_call_id: str) -> None:
        """标记尚未开始的工具调用为运行中。

        作用:
            在执行器真正调用工具前保存执行起点，供中断恢复区分未执行和结果尚未确认。
            本函数只记录状态，不执行工具本身。

        参数:
            step_id: 与 tool_call_id 共同定位调用的步骤 ID。
            tool_call_id: 模型返回的调用 ID；对应记录须为 pending，且尚无结果消息。

        返回:
            无返回值(None)；成功时已提交 running 状态和开始时间。

        异常:
            ValueError: 调用状态不是 "pending"，或 result_message_id 已有值。

        行为说明:
            1. 在事务内按组合 ID 查询调用并检查其尚未开始、尚无结果。
            2. 保存 running 和 started_at 后提交；调用方随后才能实际执行工具。
               此处不额外检查所属步骤或回合的运行状态。
        """
        with self._transaction():
            call = self._one(
                "SELECT * FROM session_tool_calls WHERE step_id = ? AND tool_call_id = ?",
                (step_id, tool_call_id),
                "工具调用",
            )
            if call["status"] != "pending" or call["result_message_id"] is not None:
                raise ValueError("工具调用已经开始或结束，不能重复执行")
            self.connection.execute(
                """UPDATE session_tool_calls SET status = 'running', started_at = ?
                    WHERE step_id = ? AND tool_call_id = ?""",
                (_now(), step_id, tool_call_id),
            )

    def _insert_tool_result(self, step_id: str, payload: dict, status: str | None) -> str:
        """保存工具结果到当前事务并关联原调用。

        作用:
            为正常工具完成和中断修复共用结果写入逻辑，生成结果消息并更新调用状态。
            调用方负责事务，函数本身不提交，也不执行工具。

        参数:
            step_id: 与 payload.tool_call_id 共同定位原调用的步骤 ID。
            payload: 可 JSON 序列化、role="tool" 的完整消息，包含非空字符串 tool_call_id。
                例如 {"role": "tool", "tool_call_id": "call_1", "content": "结果"}。
                原调用必须存在且 result_message_id 为空。
            status: 可选非空状态字符串；None 时依据 content 中的 error 字段推断。

        返回:
            新生成结果消息的内部 ID(str)。

        异常:
            ValueError: 原调用的 result_message_id 已有值，不能重复保存结果。

        行为说明:
            1. 编码消息并定位原调用，拒绝重复结果；不要求调用或步骤仍为 running。
            2. 未指定状态时尝试解码 content；字典中 error 非 None 判为 failed，否则为 completed。
               content 不是有效 JSON 时继续按原值判断，不因此拒绝结果。
            3. 保存结果消息及其调用 ID，再回填原调用的结果消息 ID、状态和完成时间。
        """
        encoded = _payload(payload, "tool")
        call_id = payload.get("tool_call_id")
        _nonempty(call_id, "工具调用 ID")
        call = self._one(
            "SELECT * FROM session_tool_calls WHERE step_id = ? AND tool_call_id = ?",
            (step_id, call_id),
            "工具调用",
        )
        if call["result_message_id"] is not None:
            raise ValueError("工具调用已经保存结果，不能重复追加")
        step = self._step(step_id)
        if status is None:
            content = payload.get("content")
            try:
                content = json.loads(content) if isinstance(content, str) else content
            except json.JSONDecodeError:
                pass
            status = "failed" if isinstance(content, dict) and content.get("error") is not None else "completed"
        status = _nonempty(status, "工具结果状态")
        message_id = self._insert_message(
            step["session_id"],
            "tool",
            encoded,
            turn_id=step["turn_id"],
            step_id=step_id,
            tool_call_id=call_id,
        )
        self.connection.execute(
            """UPDATE session_tool_calls SET result_message_id = ?, status = ?, finished_at = ?
                WHERE step_id = ? AND tool_call_id = ?""",
            (message_id, status, _now(), step_id, call_id),
        )
        return message_id

    def append_tool_result(self, step_id: str, payload: dict, *, status: str | None = None) -> MessageRecord:
        """保存工具结果消息并原子提交其调用关联。

        作用:
            在工具执行器返回后提交结果，使历史消息和调用完成状态保持一致。
            正常执行与失败结果都通过此入口保存。

        参数:
            step_id: 与 payload.tool_call_id 共同定位调用的步骤 ID。
            payload: 可 JSON 序列化、role="tool" 的完整消息，须有非空字符串 tool_call_id。
                原调用必须存在且尚未保存结果；内容结构由工具执行器提供。
            status: 可选非空状态字符串，None 时由结果中的 error 字段推断成功或失败。

        返回:
            已保存的工具结果消息(MessageRecord)。

        行为说明:
            1. 在一个事务内调用 _insert_tool_result，同时写入消息并更新原调用。
            2. 提交后读取消息记录；该读取失败不会撤销已经提交的结果。
        """
        with self._transaction():
            message_id = self._insert_tool_result(step_id, payload, status)
        return self._get_message(message_id)

    def finish_step(
        self,
        step_id: str,
        status: str,
        *,
        usage: Any = None,
        finish_reason: str | None = None,
        duration_ms: float | None = None,
        error: Any = None,
    ) -> None:
        """保存步骤状态及本次模型交互的统计信息。

        作用:
            在模型响应或步骤结束后更新持久化记录，供恢复和执行过程查看使用。
            可以用 running 保存中间信息，不代表每次调用都结束步骤。

        参数:
            step_id: 已存在的步骤 ID，不要求原状态为 running。
            status: 去除首尾空白后非空的状态字符串；不校验状态枚举或状态迁移。
            usage: 可选用量记录，须能被 _json 编码，允许数据类实例。
            finish_reason: 可选模型或执行结束原因，原样保存。
            duration_ms: 可选耗时，单位毫秒；本函数不校验非负性。
            error: 可选错误详情，须能被 _json 编码。

        返回:
            无返回值(None)；成功时状态及统计信息已经提交。

        行为说明:
            1. 校验状态，在事务内读取步骤并编码统计及错误信息。
            2. 覆盖状态和相关字段；running 将 finished_at 设为空，其他状态记录当前时间。
            3. 刷新所属会话活动时间后提交；本函数不检查工具是否全部完成。
        """
        status = _nonempty(status, "步骤状态")
        with self._transaction():
            step = self._step(step_id)
            self.connection.execute(
                """UPDATE session_steps SET status = ?, usage = ?, finish_reason = ?, duration_ms = ?,
                    error = ?, finished_at = ? WHERE step_id = ?""",
                (
                    status,
                    _json(usage) if usage is not None else None,
                    finish_reason,
                    duration_ms,
                    _json(error) if error is not None else None,
                    None if status == "running" else _now(),
                    step_id,
                ),
            )
            self._touch(step["session_id"])

    def finish_turn(
        self,
        turn_id: str,
        status: str,
        end_reason: str | None,
        *,
        retain_input: bool = True,
        error: Any = None,
    ) -> None:
        """保存回合结束记录并按需撤回上下文中的用户输入。

        作用:
            在 Runtime 结束一轮处理时保存状态、原因和错误详情。
            需要撤回失败输入时只改变上下文标记，仍保留原消息供历史查看。

        参数:
            turn_id: 已存在的回合 ID，不要求原状态为 running。
            status: 去除首尾空白后非空的状态字符串；调用方应提供终态，函数不检查状态枚举。
            end_reason: 回合结束原因，可为 None。
            retain_input: 是否保留输入用于后续模型上下文，默认 True。
                False 只允许用于尚无助手消息的回合；True 不会重新启用已撤回输入。
            error: 可选错误详情，须能被 _json 编码。

        返回:
            无返回值(None)；成功时回合信息及可选的输入撤回操作已经提交。

        异常:
            ValueError: retain_input=False，但本轮已经保存了助手消息。

        行为说明:
            1. 校验状态并在事务内读取回合，必要时确认没有助手消息再将用户输入标为非上下文。
            2. 保存状态、原因、错误和完成时间，再刷新会话活动时间，一并提交。
            3. 不联动结束步骤或工具调用；这些执行记录由各自的写入入口管理。
        """
        status = _nonempty(status, "回合状态")
        with self._transaction():
            turn = self._one("SELECT * FROM session_turns WHERE turn_id = ?", (turn_id,), "回合")
            if not retain_input:
                if self._rows("SELECT 1 FROM session_messages WHERE turn_id = ? AND role = 'assistant'", (turn_id,)):
                    raise ValueError("已经保存助手消息的回合不能撤回用户输入")
                self.connection.execute(
                    "UPDATE session_messages SET in_context = 0 WHERE message_id = ?",
                    (turn["user_message_id"],),
                )
            self.connection.execute(
                """UPDATE session_turns SET status = ?, end_reason = ?, error = ?, finished_at = ?
                    WHERE turn_id = ?""",
                (status, end_reason, _json(error) if error is not None else None, _now(), turn_id),
            )
            self._touch(turn["session_id"])

    def inspect(self, session_id: str) -> dict:
        """返回完整关联记录；展示层负责图片等内容的脱敏。"""
        session = self.get_session(session_id)
        session_id = session.session_id
        turns = self._rows("SELECT * FROM session_turns WHERE session_id = ? ORDER BY created_at, rowid", (session_id,))
        steps = self._rows(
            """SELECT s.* FROM session_steps s JOIN session_turns t ON t.turn_id = s.turn_id
                WHERE t.session_id = ? ORDER BY t.created_at, t.rowid, s.step_index""",
            (session_id,),
        )
        messages = self._rows("SELECT * FROM session_messages WHERE session_id = ? ORDER BY seq", (session_id,))
        calls = self._rows(
            """SELECT c.* FROM session_tool_calls c JOIN session_messages m ON m.message_id = c.assistant_message_id
                WHERE m.session_id = ? ORDER BY m.seq, c.rowid""",
            (session_id,),
        )
        for row in turns + steps:
            for field in ("usage", "error"):
                if field in row and row[field] is not None:
                    row[field] = json.loads(row[field])
        for row in messages:
            row["payload"] = json.loads(row["payload"])
            row["in_context"] = bool(row["in_context"])
        return {"session": asdict(session), "turns": turns, "steps": steps, "messages": messages, "tool_calls": calls}

    def recover_session(self, session_id: str) -> int:
        """修补指定会话中遗留的中断执行记录。

        作用:
            在 Runtime 显式恢复历史会话时补齐工具结果说明并结束遗留运行状态。
            只修复持久记录，不重新请求模型，也不重新执行工具。

        参数:
            session_id: 已存在会话的完整 ID；调用方应确认没有仍在实际执行的回合。
                本函数不检测其他进程是否正在使用该会话。

        返回:
            本次处理的回合数量(int)，包含遗留 running 状态或缺少工具结果的回合。
            无需修补时返回 0。

        行为说明:
            1. 检查会话并在一个事务内查找运行中的回合、运行中的步骤或缺结果的调用。
            2. 为 pending 调用补入 not_executed 说明，其余缺结果调用补入 interrupted 说明；
               将 running 步骤改为 interrupted，不尝试确认或重做工具的实际副作用。
            3. 结束 running 回合；其中没有助手消息的回合将输入移出上下文。
               有修补时刷新活动时间，全部写入一并提交后返回处理数量。
        """
        self._session_info(session_id)
        with self._transaction():
            turns = self._rows(
                """SELECT t.* FROM session_turns t WHERE t.session_id = ? AND
                    (t.status = 'running' OR EXISTS (SELECT 1 FROM session_steps s WHERE s.turn_id = t.turn_id
                    AND (s.status = 'running' OR EXISTS (SELECT 1 FROM session_tool_calls c
                    WHERE c.step_id = s.step_id AND c.result_message_id IS NULL)))) ORDER BY t.created_at, t.rowid""",
                (session_id,),
            )
            for turn in turns:
                calls = self._rows(
                    """SELECT c.* FROM session_tool_calls c JOIN session_steps s ON s.step_id = c.step_id
                        WHERE s.turn_id = ? AND c.result_message_id IS NULL ORDER BY s.step_index, c.rowid""",
                    (turn["turn_id"],),
                )
                for call in calls:
                    started = call["status"] != "pending"
                    status = "interrupted" if started else "not_executed"
                    message = (
                        "上次执行在工具结果保存前中断，结果尚未确认；请先查询实际状态，不要直接重试可能有副作用的操作。"
                        if started
                        else "上次回合中断，此工具调用尚未开始执行。"
                    )
                    self._insert_tool_result(
                        call["step_id"],
                        {
                            "role": "tool",
                            "tool_call_id": call["tool_call_id"],
                            "content": _json({"error": {"type": status, "message": message}}),
                        },
                        status,
                    )
                self.connection.execute(
                    """UPDATE session_steps SET status = 'interrupted', finished_at = ?,
                        finish_reason = 'recovered_interruption' WHERE turn_id = ? AND status = 'running'""",
                    (_now(), turn["turn_id"]),
                )
                if turn["status"] == "running":
                    if not self._rows(
                        "SELECT 1 FROM session_messages WHERE turn_id = ? AND role = 'assistant'",
                        (turn["turn_id"],),
                    ):
                        self.connection.execute(
                            "UPDATE session_messages SET in_context = 0 WHERE message_id = ?",
                            (turn["user_message_id"],),
                        )
                    self.connection.execute(
                        """UPDATE session_turns SET status = 'interrupted', end_reason = 'recovered_interruption',
                            finished_at = ? WHERE turn_id = ?""",
                        (_now(), turn["turn_id"]),
                    )
            if turns:
                self._touch(session_id)
        return len(turns)
