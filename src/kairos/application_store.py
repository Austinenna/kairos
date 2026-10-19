"""持久保存统一应用请求；每次操作在调用线程内独立打开和关闭 SQLite。"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def encode(value: object) -> str:
    """编码可持久对象并保持摘要计算稳定。

    作用:
        为请求比较、结果及通知提供相同 JSON 规则。
    参数:
        value: JSON 可表达的数据。
    返回:
        排序且不接受非有限数值的 JSON 字符串。
    行为说明:
        1. 不使用 repr 或字符串兜底序列化任意对象。
    """
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class ApplicationStore:
    """保存幂等请求和受控通知，不共享任何活动数据库连接。"""

    def __init__(self, path: Path):
        """准备私有请求库。

        作用:
            在应用服务启动时建立通用请求和通知表。
        参数:
            path: 专用数据库文件；测试使用临时目录。
        返回:
            None。
        异常:
            ValueError: 数据库是符号链接。
        行为说明:
            1. 新目录为 700，新文件为 600，不更改已有路径权限。
            2. 建表连接在当前线程关闭，实例只保留路径。
        """
        self.path = Path(path).expanduser()
        if self.path.is_symlink():
            raise ValueError("应用请求库不能是符号链接")
        missing = []
        parent = self.path.parent
        while not parent.exists():
            missing.append(parent)
            parent = parent.parent
        for directory in reversed(missing):
            directory.mkdir(mode=0o700, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS application_requests (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL,
                    turn_id TEXT,
                    user_message_id TEXT,
                    result TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    last_input_retained INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS application_requests_turn
                    ON application_requests(session_id, turn_id);
                CREATE TABLE IF NOT EXISTS application_notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    session_id TEXT,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS application_metadata (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                """
            )

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """在当前线程建立一次短连接和事务边界。

        作用:
            接收方、查询方与执行方可以使用同一个 Store 对象而不跨线程使用连接。
        返回:
            仅在上下文内部可用的 SQLite 连接。
        异常:
            捕获的原异常: 操作失败时回滚后重新抛出。
        行为说明:
            1. 成功提交，失败回滚，最终始终关闭连接。
        """
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _record(row: sqlite3.Row | None) -> dict | None:
        """解码一条请求记录。

        作用:
            向应用层返回独立数据，不返回连接绑定对象。
        参数:
            row: 查询结果或 None。
        返回:
            已解码字典或 None。
        行为说明:
            1. 结果和载荷每次重新解码，修改副本不会写回数据库。
        """
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        result["result"] = json.loads(result["result"]) if result["result"] is not None else None
        result["last_input_retained"] = bool(result["last_input_retained"])
        return result

    def get(self, request_id: str) -> dict | None:
        """读取一条请求，不触发执行。

        作用:
            支持入口恢复和重复提交后的状态查询。
        参数:
            request_id: 客户端稳定标识。
        返回:
            完整请求记录；不存在时为 None。
        行为说明:
            1. 不根据文字相同寻找其他请求。
        """
        with self.connection() as connection:
            return self._record(
                connection.execute("SELECT * FROM application_requests WHERE request_id = ?", (request_id,)).fetchone()
            )

    def submit(self, request_id: str, source: str, operation: str, session_id: str, payload: dict) -> tuple[dict, bool]:
        """原子登记请求，拒绝复用 ID 改写来源或任务。

        作用:
            在排队和模型执行前建立持久幂等边界。
        参数:
            request_id: 一次操作的稳定标识。
            source: 稳定的入口身份。
            operation: 白名单操作名称。
            session_id: 已冻结的目标或来源会话。
            payload: 完整操作输入。
        返回:
            请求记录及是否首次插入。
        异常:
            ValueError: 标识为空，或同 ID 对应不同来源、会话、操作或载荷。
        行为说明:
            1. 同 ID 同任务返回原记录；检查与插入处于一个写事务。
        """
        if any(
            not isinstance(value, str) or not value.strip() for value in (request_id, source, operation, session_id)
        ):
            raise ValueError("请求 ID、来源、操作和会话必须是非空字符串")
        serialized = encode(payload)
        digest = hashlib.sha256(encode([source, operation, session_id, payload]).encode()).hexdigest()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = self._record(
                connection.execute("SELECT * FROM application_requests WHERE request_id = ?", (request_id,)).fetchone()
            )
            if existing is not None:
                if existing["digest"] != digest:
                    raise ValueError("请求 ID 已用于另一来源、会话或内容，不能重新绑定")
                return existing, False
            connection.execute(
                """INSERT INTO application_requests
                   (request_id, source, operation, session_id, digest, payload, status)
                   VALUES (?, ?, ?, ?, ?, ?, 'queued')""",
                (request_id, source, operation, session_id, digest, serialized),
            )
            return self._record(
                connection.execute("SELECT * FROM application_requests WHERE request_id = ?", (request_id,)).fetchone()
            ), True

    def start(self, request_id: str) -> bool:
        """在产生执行副作用前标记请求已开始。

        作用:
            将可以首次执行的 queued 与恢复时不应重跑的 running 区分。
        参数:
            request_id: 待领取请求。
        返回:
            是否成功从 queued 领取。
        行为说明:
            1. 已完成或已开始的请求不会重新进入 running。
        """
        with self.connection() as connection:
            return (
                connection.execute(
                    "UPDATE application_requests SET status = 'running' WHERE request_id = ? AND status = 'queued'",
                    (request_id,),
                ).rowcount
                == 1
            )

    def cancel_queued(self, request_id: str) -> bool:
        """原子取消尚未被执行线程领取的请求。

        作用:
            排队入口取消不必等待另一个入口的长模型调用。
        参数:
            request_id: 要取消的稳定请求 ID。
        返回:
            是否从 queued 成功改为 cancelled。
        行为说明:
            1. 与 start 使用同一行条件竞争；running 请求只能合作取消。
        """
        with self.connection() as connection:
            return (
                connection.execute(
                    """UPDATE application_requests SET status = 'cancelled', error_code = 'cancelled',
                   error_message = '请求在执行前取消。', last_input_retained = 0
                   WHERE request_id = ? AND status = 'queued'""",
                    (request_id,),
                ).rowcount
                == 1
            )

    def link_turn(self, request_id: str, source: dict) -> None:
        """同步保存 Runtime 刚建立的精确回合关联。

        作用:
            必须在首次模型或工具调用前返回成功，避免恢复时猜测最近回合。
        参数:
            request_id: 当前 running 请求。
            source: Runtime 提供的 session_id、turn_id、user_message_id。
        返回:
            None。
        异常:
            RuntimeError: 请求状态或会话不匹配，关联无法确认。
        行为说明:
            1. 事务提交之后才向 Runtime 返回。
        """
        with self.connection() as connection:
            changed = connection.execute(
                """UPDATE application_requests SET turn_id = ?, user_message_id = ?
                   WHERE request_id = ? AND session_id = ? AND status = 'running' AND turn_id IS NULL""",
                (source["turn_id"], source["user_message_id"], request_id, source["session_id"]),
            ).rowcount
            if changed != 1:
                raise RuntimeError("应用请求与 Runtime 回合关联未能保存")

    def finish(
        self,
        request_id: str,
        status: str,
        *,
        result: dict | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        retained: bool = False,
    ) -> None:
        """保存终态与展示结果。

        作用:
            为完成后重复查询和失败恢复提供原结果，不再执行任务。
        参数:
            request_id: 目标请求。
            status: completed、failed、cancelled 或 interrupted。
            result: 安全展示结果，不包含完整模型请求或图片载荷。
            error_code: 受控错误类型。
            error_message: 可安全展示的简短说明。
            retained: 原输入是否保留在 Runtime 上下文。
        返回:
            None。
        异常:
            ValueError: 终态不合法。
        行为说明:
            1. 不覆盖已经保存的终态。
        """
        if status not in {"completed", "failed", "cancelled", "interrupted"}:
            raise ValueError("应用请求终态无效")
        with self.connection() as connection:
            connection.execute(
                """UPDATE application_requests SET status = ?, result = ?, error_code = ?, error_message = ?,
                   last_input_retained = ? WHERE request_id = ? AND status IN ('queued', 'running')""",
                (
                    status,
                    encode(result) if result is not None else None,
                    error_code,
                    error_message,
                    int(retained),
                    request_id,
                ),
            )

    def unfinished(self) -> list[dict]:
        """读取上次留下的任务顺序。

        作用:
            启动恢复时分别处理未开始和已经开始的请求。
        返回:
            按接收顺序排列的 queued/running 记录。
        行为说明:
            1. 本方法不改变状态。
        """
        with self.connection() as connection:
            return [
                self._record(row)
                for row in connection.execute(
                    "SELECT * FROM application_requests WHERE status IN ('queued', 'running') ORDER BY sequence"
                )
            ]

    def remember_template(self, session_id: str) -> None:
        """登记内部默认模板会话。

        作用:
            让应用界面始终隐藏用于保留初始配置的空会话。
        参数:
            session_id: 不能作为客户端聊天目标的模板 ID。
        返回:
            None。
        行为说明:
            1. 不写入模板会话正文或修改业务配置。
        """
        with self.connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO application_metadata VALUES (?, ?)", ("template:" + session_id, session_id)
            )

    def templates(self) -> set[str]:
        """读取需要从客户端列表排除的内部模板。

        作用:
            跨应用重启保持模板隐藏。
        返回:
            模板 Session ID 集合。
        行为说明:
            1. 只读取应用元数据。
        """
        with self.connection() as connection:
            return {
                row["value"]
                for row in connection.execute("SELECT value FROM application_metadata WHERE key LIKE 'template:%'")
            }

    def notify(self, event: dict) -> None:
        """保存一条后台记忆完成通知并关联真实来源。

        作用:
            让多个客户端独立读取，而不是竞争 Runtime 的单次 drain 队列。
        参数:
            event: Runtime 已生成的受控记忆事件。
        返回:
            None。
        行为说明:
            1. 按精确 Session/Turn 找到来源；历史任务没有请求关联时按会话保留。
        """
        source = event.get("source") or {}
        with self.connection() as connection:
            row = connection.execute(
                "SELECT source FROM application_requests WHERE session_id = ? AND turn_id = ? LIMIT 1",
                (source.get("session_id"), source.get("turn_id")),
            ).fetchone()
            connection.execute(
                "INSERT INTO application_notifications(source, session_id, payload) VALUES (?, ?, ?)",
                (row["source"] if row else None, source.get("session_id"), encode(event)),
            )

    def notifications(self, source: str, session_id: str, after: int) -> tuple[list[dict], int]:
        """按客户端游标读取自己的通知。

        作用:
            一个客户端取走通知不会使其他客户端丢失结果。
        参数:
            source: 当前入口身份。
            session_id: 用于兼容没有应用请求关联的历史来源。
            after: 本客户端已经读取的通知位置。
        返回:
            事件列表与新游标。
        行为说明:
            1. 不删除其他客户端尚未读取的通知。
        """
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT id, payload FROM application_notifications WHERE id > ?
                   AND (source = ? OR (source IS NULL AND session_id = ?)) ORDER BY id""",
                (after, source, session_id),
            ).fetchall()
            return [json.loads(row["payload"]) for row in rows], rows[-1]["id"] if rows else after
