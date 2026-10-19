"""把一次运行的步骤和消息写入会话存储，不参与模型协议转换。"""

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from .pricing import estimate_cost_snapshot
from .sessions import SessionStore


class TurnRecorder:
    """由 Runtime 绑定回合，由 Agent 循环在实际执行边界调用。"""

    def __init__(self, store: SessionStore, session_id: str, turn_id: str, model: str) -> None:
        """绑定会话与回合，初始化执行记录器。

        作用:
            Runtime 创建回合后建立此对象，Agent 循环随后通过它保存步骤和消息。
            此处仅初始化内存状态，数据库写入发生在后续记录方法中。

        参数:
            store: 承担持久化的 SessionStore。
            session_id: 已创建的会话 ID；应与 turn_id 所属会话一致，由调用方保证。
            turn_id: 已创建的回合 ID；本方法不查询数据库验证归属。
            model: 本轮模型名称，后续写入步骤记录。

        返回:
            无返回值（None），新实例开始时没有当前步骤。

        行为说明:
            1. 保存存储依赖及会话、回合身份。
            2. 初始化步骤列表、用量、计时和故障标记，等待循环创建首个步骤。
        """

        self.store = store
        self.session_id = session_id
        self.turn_id = turn_id
        self.model = model
        self.step_id: str | None = None
        self.step_ids: list[str] = []
        self.storage_failed = False
        self._started = 0.0
        self._finished = True
        self._usage: dict | None = None
        self._cost_snapshot: dict | None = None
        self._priced_at: datetime | None = None
        self._finish_reason: str | None = None

    @property
    def ids(self) -> dict[str, str]:
        """生成当前执行位置的关联 ID 副本。

        作用:
            为日志和观察事件提供会话、回合及当前步骤身份，无数据库副作用。

        参数:
            无显式参数；读取实例绑定的 session_id、turn_id 和当前 step_id。

        返回:
            关联 ID 字典（dict[str, str]）。未开始步骤时不包含 step_id。

        行为说明:
            1. 新建包含会话与回合 ID 的字典。
            2. 已有步骤时补入 step_id；修改返回字典不会改变记录器身份。
        """

        values = {"session_id": self.session_id, "turn_id": self.turn_id}
        if self.step_id:
            values["step_id"] = self.step_id
        return values

    def start_step(self, index: int) -> None:
        """保存新步骤并切换记录器的当前步骤。

        作用:
            Agent 循环在请求模型前调用，将步骤写入数据库，并准备本步的统计状态。

        参数:
            index: 回合内步骤序号，必须为大于等于 1 的整数，且不能与已有步骤重复。

        返回:
            无返回值（None）；新的 step_id 保存在实例中，并追加到 step_ids。

        行为说明:
            1. 由存储层生成步骤 ID 并提交 running 记录。
            2. 保存成功后才切换当前步骤，重置用量、费用、停止原因和 UTC 请求起点。
        """

        step = self._save(self.store.start_step, self.turn_id, index, self.model)
        self.step_id = step.step_id
        self.step_ids.append(step.step_id)
        self._started = perf_counter()
        self._priced_at = datetime.now(UTC)
        self._finished = False
        self._usage = None
        self._cost_snapshot = None
        self._finish_reason = None

    def record_response(self, response: dict, duration_ms: float) -> None:
        """保存模型响应的用量和停止原因。

        作用:
            模型生成结束后记录统计信息，供后续步骤收尾使用；会更新数据库。

        参数:
            response: 模型归一化响应字典，读取可选的 usage、finish_reason 字段。
                例如 {"usage": {"prompt_tokens": 10}, "finish_reason": "tool_calls"}。
            duration_ms: 本次模型请求耗时，单位为毫秒。

        返回:
            无返回值（None）；统计信息同时保存在记录器和步骤记录中。

        行为说明:
            1. 从响应中取得统计字段，缺少时使用 None。
            2. 按步骤的 UTC 请求起点固定费用快照，与用量原子保存。
            3. 将步骤状态保持为 running 写入；后面可能还有工具执行，暂不记录结束时间。
        """

        step_id = self._current_step()
        self._usage = response.get("usage")
        self._cost_snapshot = (
            estimate_cost_snapshot(self.model, self._usage, self._priced_at) if self._priced_at is not None else None
        )
        self._finish_reason = response.get("finish_reason")
        self._save(
            self.store.finish_step,
            step_id,
            "running",
            usage=self._usage,
            cost_snapshot=self._cost_snapshot,
            finish_reason=self._finish_reason,
            duration_ms=duration_ms,
        )

    def append_assistant(self, message: dict) -> str:
        """保存当前步骤的助手消息及其工具调用。

        作用:
            将完整模型回复交给存储层，建立消息与调用关联，作为开始执行工具前的保存边界。

        参数:
            message: 完整助手消息，role 必须为 assistant，整个对象应可 JSON 序列化。
                可包含 tool_calls，例如 {"role": "assistant", "content": "回答"}。
                当前步骤必须处于 running 状态，且尚未保存助手消息。

        返回:
            新助手消息的内部 ID（str），由 Kairos 生成。

        行为说明:
            1. 取得当前步骤 ID，并委托存储层原子写入消息和整批调用。
            2. 返回消息 ID；此方法只保存调用记录，真正执行由 Agent 循环负责。
        """

        return self._save(self.store.append_assistant, self._current_step(), message).message_id

    def start_tool(self, tool_call_id: str) -> None:
        """标记当前步骤中的工具调用已经开始。

        作用:
            在实际调用工具前持久保存 running 状态，供恢复流程区分未执行和结果未知。

        参数:
            tool_call_id: 模型返回的调用 ID，须对应当前步骤中 pending 且没有结果的调用。

        返回:
            无返回值（None）；调用记录变为 running，并保存开始时间。

        行为说明:
            1. 以当前 step_id 和原始 tool_call_id 定位调用。
            2. 提交开始状态后返回，实际工具执行发生在调用方。
        """

        self._save(self.store.start_tool, self._current_step(), tool_call_id)

    def append_tool_result(self, message: dict, *, status: str | None = None) -> str:
        """保存工具结果并配对当前步骤中的调用。

        作用:
            工具执行完成或中断补齐结果时调用，向会话历史追加结果并更新调用记录。

        参数:
            message: 完整工具结果消息，包含 role="tool"、tool_call_id 和 content。
                例如 {"role": "tool", "tool_call_id": "call_1", "content": "[]"}。
            status: 可选的非空状态字符串；None 时由存储层根据结果中的 error 字段判定。

        返回:
            新结果消息的内部 ID（str）。

        行为说明:
            1. 以当前步骤和消息携带的原调用 ID 交给存储层配对。
            2. 结果消息及调用关联一并提交后，返回内部消息 ID。
        """

        return self._save(self.store.append_tool_result, self._current_step(), message, status=status).message_id

    def finish_step(self, status: str, *, error: str | None = None) -> None:
        """保存当前步骤的结束状态与总耗时。

        作用:
            Agent 循环在步骤完成或中断时收尾，使用已记录的模型用量和停止原因更新数据库。

        参数:
            status: 要保存的非空状态字符串，例如 completed、failed 或 cancelled。
            error: 可选的错误说明；调用方应先移除不适合展示的内容。

        返回:
            无返回值（None）。步骤已收尾或已有保存故障时直接返回，不再次写库。

        行为说明:
            1. 先检查是否已经收尾或发生保存故障，避免重复写入或覆盖待恢复状态。
            2. 计算总耗时，连同模型用量、原费用快照和结束状态提交，不按收尾时间重算费用。
            3. 保存成功后标记本步骤已收尾。
        """

        if self._finished or self.storage_failed:
            return
        self._save(
            self.store.finish_step,
            self._current_step(),
            status,
            usage=self._usage,
            cost_snapshot=self._cost_snapshot,
            finish_reason=self._finish_reason,
            duration_ms=round((perf_counter() - self._started) * 1000, 3),
            error=error,
        )
        self._finished = True

    def _save(self, operation: Callable, *args: Any, **kwargs: Any) -> Any:
        """执行存储操作并标记 SQLite 保存故障。

        作用:
            作为记录器调用存储层的统一入口，使 Runtime 能识别需要显式恢复的保存失败。

        参数:
            operation: 将要调用的存储函数。
            *args: 原样传给该函数的位置参数。
            **kwargs: 原样传给该函数的关键字参数。

        返回:
            被调用函数的原返回值（Any），包括可能的 None。

        异常:
            sqlite3.Error（显式重新抛出）: operation 抛出的 SQLite 异常被本函数捕获后，
                先设置 storage_failed=True，再通过裸 raise 重新抛出原异常。

        行为说明:
            1. 使用原参数调用存储函数，不自行开启或提交事务。
            2. 仅在 SQLite 异常时设置故障标记，保留异常供上层处理。
        """

        try:
            return operation(*args, **kwargs)
        except sqlite3.Error:
            self.storage_failed = True
            raise

    def _current_step(self) -> str:
        """取得当前已经创建的步骤 ID。

        作用:
            为消息和工具记录方法提供归属 ID，阻止在步骤创建前写入这些记录。

        参数:
            无显式参数；读取实例的 step_id。

        返回:
            当前步骤 ID（str）。

        异常:
            RuntimeError: step_id 为 None，尚未开始步骤。

        行为说明:
            1. 检查当前步骤是否已经建立。
            2. 返回已有 ID；不生成新 ID，也不查询或修改数据库。
        """

        if self.step_id is None:
            raise RuntimeError("尚未开始执行步骤")
        return self.step_id
