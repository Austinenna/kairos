"""同进程多入口应用层：一个固定线程拥有 Runtime，入口只操作各自会话视图。"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import RLock, Thread, current_thread
from typing import Any
from uuid import uuid4

from .application_store import ApplicationStore
from .cancellation import CancellationToken
from .legacy_images import MAX_SESSION_BASE64_BYTES, MAX_SESSION_IMAGES, ImageInput, ImageInputError, image_budget_usage
from .pricing import Usage, usage_from_normalized
from .runtime import AgentRuntime, AgentTurnResult
from .sessions import SessionInfo
from .usage_stats import SessionUsage, summarize_usage


class ApplicationRequestError(RuntimeError):
    """表示已经保存终态的请求失败；重复提交不会重新执行。"""

    def __init__(self, record: dict):
        """保留可查询的请求身份和安全错误。

        作用:
            将执行线程的失败转换成入口可展示且可恢复的错误。
        参数:
            record: 请求库中的终态记录。
        返回:
            None。
        行为说明:
            1. 不携带原始 SDK 异常、凭据或模型请求。
        """
        self.request_id = record["request_id"]
        self.status = record["status"]
        self.error_code = record.get("error_code")
        self.last_input_retained = record.get("last_input_retained", False)
        super().__init__(record.get("error_message") or "请求未完成；不会自动重新执行，请检查会话记录。")


def _usage(value: dict | None) -> Usage | None:
    """恢复显示层用量对象。

    作用:
        保持持久结果中未知计数为 None。
    参数:
        value: 公开用量字段或 None。
    返回:
        Usage 或 None。
    行为说明:
        1. 不重算费用或猜测缺失计数。
    """
    return Usage(**value) if value is not None else None


def _summary(value: dict | None) -> SessionUsage | None:
    """恢复持久会话用量快照。

    作用:
        让初次完成与重复查询具有相同展示类型。
    参数:
        value: SessionUsage 的公开字段或 None。
    返回:
        SessionUsage 或 None。
    行为说明:
        1. 只恢复字段，不按新价格重新计费。
    """
    return (
        SessionUsage(
            **{**value, "usage": _usage(value["usage"]), "last_request_usage": _usage(value["last_request_usage"])}
        )
        if value is not None
        else None
    )


def _result_value(envelope: dict | None) -> object:
    """把安全结果投影恢复成 CLI 所需对象。

    作用:
        兼容现有渲染接口，同时不传递原始模型历史或观察载荷。
    参数:
        envelope: 已保存的结果与类型。
    返回:
        AgentTurnResult、SessionInfo 或 None。
    行为说明:
        1. AgentTurnResult 的 messages、events、tool_results 始终为空。
    """
    if envelope is None:
        return None
    value = envelope["value"]
    if envelope["type"] == "session":
        return SessionInfo(**value)
    if envelope["type"] == "turn":
        return AgentTurnResult(
            **{
                **value,
                "usage": _usage(value["usage"]),
                "last_request_usage": _usage(value["last_request_usage"]),
                "session_usage": _summary(value["session_usage"]),
                "step_ids": tuple(value["step_ids"]),
            },
            messages=[],
            events=[],
            tool_results=[],
        )
    return value


def _pack_result(value: object, view: dict) -> dict:
    """只保存展示所需结果和固定会话快照。

    作用:
        防止 dataclasses.asdict 把完整历史、图片和记忆请求一起复制到结果。
    参数:
        value: Runtime 返回的回合、会话概要或 None。
        view: 当前入口的公开会话快照。
    返回:
        可存入请求库的 JSON 对象。
    行为说明:
        1. 显式选择回合字段；不保存原始 events 和 messages。
    """
    if isinstance(value, AgentTurnResult):
        result = {
            "content": value.content,
            "usage": asdict(value.usage) if value.usage is not None else None,
            "finish_reason": value.finish_reason,
            "session_id": value.session_id,
            "turn_id": value.turn_id,
            "step_ids": list(value.step_ids),
            "last_request_usage": asdict(value.last_request_usage) if value.last_request_usage is not None else None,
            "session_usage": asdict(value.session_usage) if value.session_usage is not None else None,
        }
        kind = "turn"
    elif isinstance(value, SessionInfo):
        result, kind = asdict(value), "session"
    else:
        result, kind = value, "value"
    return {"type": kind, "value": result, "view": deepcopy(view)}


def _display_event(event: dict) -> dict | None:
    """将循环事件裁成终端展示字段。

    作用:
        保留流式正文和边界，避免复制完整模型请求、工具参数及记忆。
    参数:
        event: Runtime 产生的观察或正文事件。
    返回:
        展示事件；不需要展示的类型返回 None。
    行为说明:
        1. 字段采用白名单，model_request 等只保留身份与边界。
    """
    kind = event.get("event")
    if kind not in {"text_delta", "model_request", "model_response", "model_error", "tool_result"}:
        return None
    keys = ("event", "session_id", "turn_id", "step_id", "request_number", "tool_round")
    result = {key: event[key] for key in keys if key in event}
    if kind == "text_delta":
        result["text"] = event.get("text", "")
    return result


@dataclass
class _Call:
    """表示一次只在工作线程执行的内部调用。"""

    operation: str
    client: RuntimeClient
    session_id: str | None
    arguments: dict
    future: Future


class _Task:
    """让同一持久请求的多个等待者共享一次执行和独立事件队列。"""

    def __init__(self, record: dict):
        """创建请求执行状态。

        作用:
            固定请求载荷、取消标记和最终完成对象。
        参数:
            record: 请求库已经接受的记录。
        返回:
            None。
        行为说明:
            1. 不执行模型，不共享 SQLite 连接。
        """
        self.record = record
        self.future: Future = Future()
        self.token = CancellationToken()
        self._lock = RLock()
        self._events: list[dict] = []
        self._listeners: list[Queue] = []
        self.last_view: dict | None = None
        self.last_input_retained = False

    def subscribe(self) -> Queue:
        """为调用线程提供独立正文队列。

        作用:
            重复等待同一请求不会抢走另一等待者的片段。
        返回:
            本等待者专用队列。
        行为说明:
            1. 先补入本次运行已有片段，再接收后续片段。
        """
        queue: Queue = Queue()
        with self._lock:
            for event in self._events:
                queue.put(deepcopy(event))
            self._listeners.append(queue)
        return queue

    def unsubscribe(self, queue: Queue) -> None:
        """释放一个已经离开的事件订阅。

        作用:
            不保留完成或分离调用者的队列。
        参数:
            queue: subscribe 返回的队列。
        返回:
            None。
        行为说明:
            1. 不取消请求，也不影响其他等待者。
        """
        with self._lock:
            if queue in self._listeners:
                self._listeners.remove(queue)

    def emit(self, event: dict) -> None:
        """只向内存队列写入受控事件。

        作用:
            避免在 Runtime 工作线程执行 Rich 或入口回调。
        参数:
            event: Runtime 原始事件。
        返回:
            None。
        行为说明:
            1. 不等待界面、网络或接收者处理。
        """
        projected = _display_event(event)
        if projected is None:
            return
        with self._lock:
            self._events.append(projected)
            for queue in self._listeners:
                queue.put(deepcopy(projected))


class ApplicationService:
    """统一拥有一个 Runtime 的生命周期，为多个入口提供串行执行。"""

    def __init__(self, runtime_factory: Callable[[], AgentRuntime], *, request_path: Path):
        """启动专有执行线程和私有请求存储。

        作用:
            只在有入口或恢复任务需要时创建业务 Runtime。
        参数:
            runtime_factory: 在 worker 内调用、返回默认新空会话 Runtime 的工厂。
            request_path: 独立应用请求库路径。
        返回:
            None。
        行为说明:
            1. 工厂、SQLite 和 Runtime 的使用与关闭均在同一个 worker 内。
            2. 请求库使用短连接；生产调用方仍须持有应用运行锁。
        """
        self.store = ApplicationStore(request_path)
        self._factory = runtime_factory
        self._queue: Queue = Queue()
        self._lock = RLock()
        self._accepting = True
        self._stop_queued = False
        self._tasks: dict[str, _Task] = {}
        self._clients: dict[str, RuntimeClient] = {}
        self._runtime: AgentRuntime | None = None
        self._epoch = str(uuid4())
        self._view_revision = 0
        self._template_id: str | None = None
        self._templates = self.store.templates()
        self._ready: Future = Future()
        self._shutdown: Future = Future()
        self._worker = Thread(target=self._run, name="kairos-application")
        self._worker.start()
        try:
            self._ready.result()
        except BaseException:
            self._worker.join()
            raise

    def client(self, source: str, session_id: str | None = None, resume_latest: bool = False) -> RuntimeClient:
        """为入口创建独立会话视图。

        作用:
            入口可以新建或精确恢复会话，不继承其他入口的当前模型配置。
        参数:
            source: 稳定来源；微信应包含账号和外部对话身份。
            session_id: 可选恢复目标，与 resume_latest 互斥。
            resume_latest: 选择最近可见会话，不选择内部默认模板。
        返回:
            只操作自己会话的 RuntimeClient。
        异常:
            ValueError: 来源为空，或恢复选项冲突。
        行为说明:
            1. 会话准备经队列完成之后才返回客户端。
        """
        if not isinstance(source, str) or not source.strip():
            raise ValueError("入口来源不能为空")
        if session_id is not None and resume_latest:
            raise ValueError("指定会话与继续最近会话不能同时使用")
        client = RuntimeClient(self, source)
        self._call(client, "attach", {"reference": session_id, "resume_latest": resume_latest})
        return client

    def _runtime_ready(self) -> AgentRuntime:
        """只在 worker 内初始化唯一 Runtime 和默认模板。

        作用:
            保存初始配置的私有空会话，避免后来新入口继承当前活动会话。
        返回:
            当前 Runtime。
        异常:
            RuntimeError: 工厂返回的默认会话已经包含用户消息。
        行为说明:
            1. 模板从不发送消息、命名或对客户端暴露。
        """
        if self._runtime is None:
            self._runtime = self._factory()
            if self._runtime.session_info.message_count:
                raise RuntimeError("应用 Runtime 工厂必须创建默认空会话")
            self._template_id = self._runtime.session_id
            self.store.remember_template(self._template_id)
            self._templates.add(self._template_id)
        return self._runtime

    def _visible_sessions(self) -> list[SessionInfo]:
        """列出可由客户端选择的真实会话。

        作用:
            防止内部模板进入列表或成为 --continue 的目标。
        返回:
            Runtime 会话列表中的非模板项。
        行为说明:
            1. 保持原有最近活动排序。
        """
        return [item for item in self._runtime_ready().list_sessions() if item.session_id not in self._templates]

    def _select(self, reference: str) -> AgentRuntime:
        """在空闲执行线程选择一个客户端会话。

        作用:
            每次执行依据请求中的固定 ID，不使用全局最近会话。
        参数:
            reference: 完整 ID 或明确恢复命令使用的唯一前缀。
        返回:
            已选择目标的 Runtime。
        异常:
            ValueError: 目标是内部模板。
        行为说明:
            1. 只有需要切换时恢复，普通同会话请求不会重跑恢复逻辑。
        """
        runtime = self._runtime_ready()
        info = runtime.session_store.get_session(reference)
        if info.session_id in self._templates:
            raise ValueError("内部默认模板不能作为聊天会话")
        if runtime.session_id != info.session_id:
            runtime.resume_session(info.session_id)
        return runtime

    def _view(self) -> dict:
        """捕获当前会话的入口展示和预算快照。

        作用:
            入口属性读取不排队等待别的长任务，也不会读到其他会话当前值。
        返回:
            不含正文、模型历史或凭据的可序列化字段。
        行为说明:
            1. 用量读取失败仅标未知；图片预算由当前实际历史计算。
        """
        runtime = self._runtime_ready()
        self._view_revision += 1
        try:
            usage = runtime.session_usage()
        except sqlite3.Error:
            usage = None
        return {
            "epoch": self._epoch,
            "revision": self._view_revision,
            "session_info": asdict(runtime.session_info),
            "session_id": runtime.session_id,
            "model_name": runtime.model_name,
            "system_prompt": runtime.system_prompt,
            "thinking_enabled": runtime.thinking_enabled,
            "supports_thinking": runtime.supports_thinking,
            "recovered_turns": runtime.recovered_turns,
            "thinking_state_corrected": runtime.thinking_state_corrected,
            "last_input_retained": runtime.last_input_retained,
            "session_usage": asdict(usage) if usage is not None else None,
            "image_budget": list(image_budget_usage(runtime.messages)),
            "memory_enabled": runtime.memory_extractor is not None,
            "memory_pending_count": runtime.memory_pending_count,
        }

    def _publish(self, view: dict) -> None:
        """刷新绑定同会话客户端的共享显示字段。

        作用:
            另一入口重命名、切换思考或添加图片后，当前入口预检仍使用较新快照。
        参数:
            view: worker 捕获的固定会话快照。
        返回:
            None。
        行为说明:
            1. 不覆盖各入口上次请求保留状态、结果或自身会话选择。
        """
        excluded = {"last_input_retained", "recovered_turns", "thinking_state_corrected"}
        shared = {key: value for key, value in view.items() if key not in excluded}
        with self._lock:
            clients = list(self._clients.values())
        for client in clients:
            with client._state_lock:
                if client._view.get("session_id") == view["session_id"]:
                    client._view.update(deepcopy(shared))

    def _call(self, client: RuntimeClient, operation: str, arguments: dict | None = None) -> Any:
        """将会话读取或客户端准备交给执行线程。

        作用:
            避免入口线程直接使用 Runtime 的连接与可变状态。
        参数:
            client: 请求入口。
            operation: 内部白名单调用名称。
            arguments: 调用参数。
        返回:
            worker 返回的独立结果。
        异常:
            RuntimeError: 服务停止接收或客户端已经分离。
        行为说明:
            1. 等待期间不占用服务锁。
        """
        with self._lock:
            if not self._accepting or client._closed:
                raise RuntimeError("应用服务或客户端已关闭")
            future: Future = Future()
            self._queue.put(_Call(operation, client, client._view.get("session_id"), arguments or {}, future))
        return future.result()

    def _run_call(self, call: _Call) -> None:
        """执行一次内部白名单操作并发布结果。

        作用:
            为入口准备会话、读取历史及统计提供唯一 Runtime owner。
        参数:
            call: 已固定参数和会话的内部调用。
        返回:
            None，结果写入 Future。
        异常:
            ValueError: 没有最近会话，或内部操作名称未知。
        行为说明:
            1. 无来源的新客户端始终先恢复默认模板再新建会话。
        """
        runtime = self._runtime_ready()
        if call.operation == "attach":
            reference = call.arguments["reference"]
            if call.arguments["resume_latest"]:
                sessions = self._visible_sessions()
                if not sessions:
                    raise ValueError("没有可恢复的会话")
                reference = sessions[0].session_id
            if reference is None:
                runtime.resume_session(self._template_id)
                runtime.new_session()
            else:
                self._select(reference)
            with self._lock:
                self._clients[call.client._identity] = call.client
            value = runtime.session_info
        else:
            runtime = self._select(call.session_id)
            methods = {
                "list_sessions": self._visible_sessions,
                "history": runtime.history,
                "session_details": runtime.session_details,
                "session_usage": runtime.session_usage,
                "wait_for_memory": runtime.wait_for_memory,
            }
            if call.operation not in methods:
                raise ValueError("未知应用读取操作")
            value = methods[call.operation](**call.arguments)
        if call.operation == "wait_for_memory":
            self._collect_memory()
        view = self._view()
        call.client._apply(view, preserve_private=call.operation != "attach")
        self._publish(view)
        call.future.set_result(deepcopy(value))

    def _submit(self, client: RuntimeClient, operation: str, payload: dict, request_id: str | None) -> _Task:
        """先持久接收请求，再加入共享执行队列。

        作用:
            重复 ID 共享当前执行或返回原结果，不重新调用模型和控制操作。
        参数:
            client: 提交入口。
            operation: 白名单操作。
            payload: 已固定输入。
            request_id: 可选稳定 ID，省略时生成 UUID。
        返回:
            可等待和取消的内部任务。
        异常:
            RuntimeError: 服务或入口已关闭。
        行为说明:
            1. 持久记录冻结来源和 Session；完成任务只恢复缓存展示结果。
        """
        with self._lock:
            if not self._accepting or client._closed:
                raise RuntimeError("应用服务或客户端已关闭")
            request_id = str(uuid4()) if request_id is None else request_id
            record, _ = self.store.submit(request_id, client.source, operation, client.session_id, payload)
            client.last_request_id = request_id
            task = self._tasks.get(request_id)
            if task is not None:
                return task
            task = _Task(record)
            if record["status"] in {"queued", "running"}:
                self._tasks[request_id] = task
                self._queue.put(task)
            else:
                self._settle(task, record, replay=True)
            return task

    @staticmethod
    def _settle(task: _Task, record: dict, *, replay: bool = False) -> None:
        """用持久终态结束等待，不再产生业务副作用。

        作用:
            初次执行和幂等查询使用同一个结果/错误分支。
        参数:
            task: 等待者共享任务。
            record: 请求终态。
            replay: 缓存回答是否需要补发完整正文事件。
        返回:
            None。
        行为说明:
            1. 失败以受控记录交给调用线程；取消在执行结束后才传递 KeyboardInterrupt。
        """
        with task._lock:
            if task.future.done():
                return
            if replay and record["status"] == "completed" and record["operation"] == "respond":
                result = _result_value(record["result"])
                task.emit({"event": "model_request", "request_number": 1})
                task.emit({"event": "text_delta", "request_number": 1, "text": result.content})
                task.emit({"event": "model_response", "request_number": 1})
            task.future.set_result(record)

    def _perform(self, task: _Task) -> object:
        """在固定会话上执行一次已领取的操作。

        作用:
            是模型与会话变更的唯一应用层入口。
        参数:
            task: 已标记 running 的任务。
        返回:
            Runtime 操作的结果。
        异常:
            ValueError: 操作名称不在白名单。
        行为说明:
            1. 新建与重置先选本请求来源会话，不继承另一入口当前状态。
        """
        row = task.record
        runtime = self._runtime_ready() if row["operation"] == "resume_session" else self._select(row["session_id"])
        task.token.raise_if_cancelled()
        payload = row["payload"]
        operation = row["operation"]
        if operation == "respond":
            return runtime.respond(
                payload["prompt"],
                images=tuple(ImageInput(**image) for image in payload["images"]),
                image_labels=payload["image_labels"],
                on_event=task.emit,
                cancel_token=task.token,
                on_turn_start=lambda source: self.store.link_turn(row["request_id"], source),
            )
        if operation == "new_session":
            return runtime.new_session(**payload)
        if operation == "reset":
            runtime.reset(**payload)
            return runtime.session_info
        if operation == "resume_session":
            info = runtime.session_store.get_session(payload["reference"])
            if info.session_id in self._templates:
                raise ValueError("内部默认模板不能作为聊天会话")
            return runtime.resume_session(info.session_id)
        if operation == "rename_session":
            return runtime.rename_session(**payload)
        if operation == "set_thinking":
            return runtime.set_thinking(**payload)
        raise ValueError("未知应用操作")

    def _run_task(self, task: _Task) -> None:
        """执行或恢复一个持久请求并保存终态。

        作用:
            将执行失败与结果保存失败分开，保存故障不会触发模型重跑。
        参数:
            task: 统一任务。
        返回:
            None。
        行为说明:
            1. queued 才能首次执行；running 只按精确 Turn 恢复。
            2. 终态保存失败留待恢复，不用一份新错误覆盖已生成答案。
        """
        request_id = task.record["request_id"]
        current = self.store.get(request_id)
        if current["status"] == "running":
            self._recover_request(current)
            self._settle(task, self.store.get(request_id), replay=True)
            return
        if current["status"] != "queued":
            self._settle(task, current, replay=True)
            return
        if task.token.is_cancelled:
            self.store.finish(request_id, "cancelled", error_code="cancelled", error_message="请求在执行前取消。")
            self._settle(task, self.store.get(request_id))
            return
        if not self.store.start(request_id):
            self._settle(task, self.store.get(request_id), replay=True)
            return
        try:
            value = self._perform(task)
        except (Exception, KeyboardInterrupt) as error:
            cancelled = isinstance(error, KeyboardInterrupt)
            owns_target = self._runtime is not None and self._runtime.session_id == current["session_id"]
            linked = self.store.get(request_id)
            retained = bool(
                owns_target
                and current["operation"] == "respond"
                and linked["turn_id"] is not None
                and self._runtime.last_input_retained
            )
            task.last_input_retained = retained
            view = None
            if owns_target:
                try:
                    view = self._view()
                    view["last_input_retained"] = retained
                    task.last_view = view
                    self._publish(view)
                except (OSError, RuntimeError, ValueError, sqlite3.Error):
                    pass
            self.store.finish(
                request_id,
                "cancelled" if cancelled else "failed",
                result=_pack_result(None, view) if view is not None else None,
                error_code="cancelled" if cancelled else type(error).__name__,
                error_message=(
                    "本轮已取消，已执行的工具不会撤销。"
                    if cancelled
                    else f"本轮执行失败（{type(error).__name__}）；未自动重试，请检查会话记录。"
                ),
                retained=retained,
            )
        else:
            task.last_input_retained = current["operation"] == "respond"
            view = self._view()
            task.last_view = view
            self._publish(view)
            self.store.finish(
                request_id, "completed", result=_pack_result(value, view), retained=view["last_input_retained"]
            )
        self._settle(task, self.store.get(request_id))

    def _recover_request(self, record: dict) -> None:
        """依据精确 Turn 恢复结果；其他已开始操作保守标记中断。

        作用:
            覆盖 Runtime 已落盘而应用结果尚未提交的崩溃窗口。
        参数:
            record: 遗留 running 请求。
        返回:
            None。
        行为说明:
            1. 不选择最近 Turn，不重新运行模型、工具或未知控制操作。
        """
        if record["operation"] == "respond" and record["turn_id"] is not None:
            runtime = self._runtime_ready()
            try:
                details = runtime.session_store.inspect(record["session_id"])
            except (KeyError, ValueError):
                details = None
            if details is not None:
                turn = next((item for item in details["turns"] if item["turn_id"] == record["turn_id"]), None)
                if turn is not None and turn["status"] == "completed":
                    messages = [
                        item
                        for item in details["messages"]
                        if item["turn_id"] == record["turn_id"] and item["role"] == "assistant"
                    ]
                    steps = [item for item in details["steps"] if item["turn_id"] == record["turn_id"]]
                    if messages and steps:
                        all_usage = runtime.session_store.usage_steps(record["session_id"])
                        last_position = max(
                            index for index, step in enumerate(all_usage) if step["turn_id"] == turn["turn_id"]
                        )
                        turn_usage = [step for step in all_usage if step["turn_id"] == turn["turn_id"]]
                        self._select(record["session_id"])
                        view = self._view()
                        view["last_input_retained"] = True
                        summary = summarize_usage(all_usage[: last_position + 1])
                        view["session_usage"] = asdict(summary)
                        result = AgentTurnResult(
                            content=messages[-1]["payload"].get("content") or "",
                            messages=[],
                            events=[],
                            tool_results=[],
                            usage=summarize_usage(turn_usage).usage,
                            finish_reason=steps[-1]["finish_reason"],
                            session_id=record["session_id"],
                            turn_id=record["turn_id"],
                            step_ids=tuple(step["step_id"] for step in steps),
                            last_request_usage=usage_from_normalized(steps[-1]["usage"]),
                            session_usage=summary,
                        )
                        self.store.finish(
                            record["request_id"], "completed", result=_pack_result(result, view), retained=True
                        )
                        return
        self.store.finish(
            record["request_id"],
            "interrupted",
            error_code="interrupted",
            error_message="上次执行结果尚未确认；没有自动重跑模型或工具，请先检查会话记录。",
            retained=True,
        )

    def _collect_memory(self) -> None:
        """集中保存后台记忆通知并刷新缓存计数。

        作用:
            任一客户端离开不会关闭提取器，通知不会被其他客户端抢走。
        返回:
            None。
        行为说明:
            1. 只有 worker 调用 Runtime drain，事件正文仍由 Runtime 的受控结构提供。
        """
        if self._runtime is None:
            return
        for event in self._runtime.drain_memory_events():
            self.store.notify(event)
        with self._lock:
            clients = list(self._clients.values())
        for client in clients:
            with client._state_lock:
                client._view["memory_pending_count"] = self._runtime.memory_pending_count

    def _run(self) -> None:
        """拥有所有 Runtime 操作与最终关闭。

        作用:
            将入口并发转换为全局串行执行，短数据库连接各自在调用线程使用。
        返回:
            None。
        行为说明:
            1. 启动时 running 只恢复结果，queued 可首次执行。
            2. 每个任务异常只结束对应 Future，不把取消扩散到下一请求。
            3. 服务结束时等待记忆并关闭 Runtime，最后发布 shutdown 完成。
        """
        try:
            for record in self.store.unfinished():
                if record["status"] == "running":
                    self._recover_request(record)
                else:
                    task = _Task(record)
                    self._tasks[record["request_id"]] = task
                    self._queue.put(task)
            self._ready.set_result(None)
            while True:
                try:
                    item = self._queue.get(timeout=0.1)
                except Empty:
                    self._collect_memory()
                    continue
                if item is None:
                    break
                try:
                    if isinstance(item, _Call):
                        self._run_call(item)
                    else:
                        self._run_task(item)
                except BaseException as error:
                    if not item.future.done():
                        item.future.set_exception(error)
                finally:
                    if isinstance(item, _Task):
                        with self._lock:
                            self._tasks.pop(item.record["request_id"], None)
                self._collect_memory()
        except BaseException as error:
            if not self._ready.done():
                self._ready.set_exception(error)
            with self._lock:
                self._accepting = False
                outstanding = list(self._tasks.values())
                while True:
                    try:
                        waiting = self._queue.get_nowait()
                    except Empty:
                        break
                    if waiting is not None:
                        outstanding.append(waiting)
            for waiting in outstanding:
                if not waiting.future.done():
                    waiting.future.set_exception(RuntimeError("应用执行线程已停止，请检查存储后重启"))
            self._shutdown.set_exception(error)
        finally:
            try:
                if self._runtime is not None:
                    self._runtime.close()
                    self._collect_memory()
            except BaseException as error:
                if not self._shutdown.done():
                    self._shutdown.set_exception(error)
            if not self._shutdown.done():
                self._shutdown.set_result(None)

    def cancel_all(self) -> None:
        """非阻塞地请求取消当前及排队任务。

        作用:
            给主进程关停提供独立控制路径，不提前关闭 Runtime。
        返回:
            None。
        行为说明:
            1. 只设置各请求的 Event，不等 worker 或访问数据库。
        """
        with self._lock:
            tasks = list(self._tasks.values())
        for task in tasks:
            task.token.cancel()

    def stop_accepting(self) -> None:
        """关闭新任务准入，保留查询和已有任务收尾。

        作用:
            入口停止后、等待入口退出前阻止迟到提交。
        返回:
            None。
        行为说明:
            1. 不关闭 Runtime，也不等待执行线程。
        """
        with self._lock:
            self._accepting = False

    def _cancel_task(self, task: _Task) -> None:
        """及时取消 queued 请求，running 请求等待合作停止。

        作用:
            取消本入口排队任务不等待或取消其他入口的模型回合。
        参数:
            task: 当前调用者自己的任务。
        返回:
            None。
        行为说明:
            1. 标记先设置；原子取消 queued 后即可结束等待。
            2. 存储不可用时仍保留标记并等待真实执行确认。
        """
        task.token.cancel()
        try:
            cancelled = self.store.cancel_queued(task.record["request_id"])
        except sqlite3.Error:
            return
        if cancelled:
            self._settle(task, self.store.get(task.record["request_id"]))

    def close(self, cancel_pending: bool = True) -> None:
        """停止接收，等待任务及后台记忆收尾后关闭服务。

        作用:
            应用唯一 owner 负责生命周期，客户端退出不执行本操作。
        参数:
            cancel_pending: 是否请求取消当前和排队请求；False 则正常处理已接受任务。
        返回:
            None。
        异常:
            RuntimeError: 从 Runtime 工作线程自身调用关闭。
        行为说明:
            1. 取消只发信号，最终仍等待真正停止和 worker 内资源关闭。
            2. 收尾期间再次 Ctrl+C 只再次请求取消，不提前放弃后台线程。
        """
        if current_thread() is self._worker:
            raise RuntimeError("不能从应用执行线程等待自身关闭")
        with self._lock:
            self._accepting = False
            if cancel_pending:
                self.cancel_all()
            if not self._stop_queued:
                self._stop_queued = True
                self._queue.put(None)
        while self._worker.is_alive():
            try:
                self._worker.join(0.1)
            except KeyboardInterrupt:
                self.cancel_all()
        self._shutdown.result()


class RuntimeClient:
    """保留一个入口自己的会话和展示快照，不拥有业务 Runtime。"""

    def __init__(self, service: ApplicationService, source: str):
        """建立尚未附着会话的客户端。

        作用:
            由 service.client 创建，来源用于幂等和通知路由。
        参数:
            service: 统一应用服务。
            source: 稳定入口身份。
        返回:
            None。
        行为说明:
            1. 属性只读本入口快照；不会触碰 worker 的当前会话。
        """
        self._service = service
        self.source = source
        self._identity = str(uuid4())
        self._state_lock = RLock()
        self._operation_lock = RLock()
        self._view: dict = {}
        self._closed = False
        self._notification_cursor = 0
        self.last_request_id: str | None = None
        self.last_result: AgentTurnResult | None = None

    def _apply(self, view: dict, *, preserve_private: bool = False) -> None:
        """安装 worker 返回的本入口快照。

        作用:
            新建、切换和请求完成后一起更新会话与用量等字段。
        参数:
            view: 公开字段字典。
            preserve_private: 只读刷新时保留本入口上一请求的保留与恢复状态。
        返回:
            None。
        行为说明:
            1. 深拷贝避免多个入口共享可变字典。
        """
        with self._state_lock:
            private_keys = {"last_input_retained", "recovered_turns", "thinking_state_corrected"}
            same_session = self._view.get("session_id") == view.get("session_id")
            stale = same_session and (
                (self._view.get("epoch") == self._service._epoch and view.get("epoch") != self._service._epoch)
                or (
                    self._view.get("epoch") == view.get("epoch")
                    and self._view.get("revision", -1) > view.get("revision", -1)
                )
            )
            if stale:
                if not preserve_private:
                    self._view.update({key: deepcopy(view[key]) for key in private_keys if key in view})
                return
            preserved = {key: self._view[key] for key in private_keys if key in self._view} if preserve_private else {}
            self._view = {**deepcopy(view), **preserved}

    def _field(self, name: str, default=None):
        """读取独立缓存字段。

        作用:
            终端编辑与渲染不等待另一入口的模型请求。
        参数:
            name: 字段名。
            default: 尚未存在时的默认值。
        返回:
            独立缓存值。
        行为说明:
            1. 不提交工作线程查询。
        """
        with self._state_lock:
            return deepcopy(self._view.get(name, default))

    session_id = property(lambda self: self._field("session_id"))
    session_info = property(lambda self: SessionInfo(**self._field("session_info")))
    model_name = property(lambda self: self._field("model_name"))
    system_prompt = property(lambda self: self._field("system_prompt"))
    thinking_enabled = property(lambda self: self._field("thinking_enabled", False))
    supports_thinking = property(lambda self: self._field("supports_thinking", False))
    recovered_turns = property(lambda self: self._field("recovered_turns", 0))
    thinking_state_corrected = property(lambda self: self._field("thinking_state_corrected", False))
    last_input_retained = property(lambda self: self._field("last_input_retained", False))
    memory_pending_count = property(lambda self: self._field("memory_pending_count", 0))
    memory_extractor = property(lambda self: True if self._field("memory_enabled", False) else None)

    def _wait(self, task: _Task, on_event: Callable[[dict], None] | None = None):
        """在调用线程泵送事件并等待真实执行结束。

        作用:
            Rich 只运行于入口线程；Ctrl+C 只取消本请求且等 worker 退出当前执行。
        参数:
            task: 已持久接受的请求。
            on_event: 入口展示回调。
        返回:
            已完成的结果。
        异常:
            KeyboardInterrupt: 调用者请求了取消或执行确认 cancelled。
            ApplicationRequestError: 请求失败或执行结果不确定。
            捕获的回调异常: 展示失败时等待业务结束后再抛出。
        行为说明:
            1. 展示回调异常不在 worker 内抛出，也不导致任务自动重试。
        """
        queue = task.subscribe()
        interrupted = False
        callback_error = None
        try:
            while not task.future.done() or not queue.empty():
                try:
                    event = queue.get(timeout=0.05)
                    if on_event is not None and not interrupted and callback_error is None:
                        try:
                            on_event(event)
                        except KeyboardInterrupt:
                            interrupted = True
                            self._service._cancel_task(task)
                        except Exception as error:
                            callback_error = error
                except Empty:
                    continue
                except KeyboardInterrupt:
                    interrupted = True
                    self._service._cancel_task(task)
            try:
                record = task.future.result()
            except BaseException:
                if task.last_view is not None:
                    self._apply(task.last_view)
                else:
                    with self._state_lock:
                        self._view["last_input_retained"] = task.last_input_retained
                raise
            if record["result"] is not None:
                self._apply(record["result"]["view"])
            else:
                with self._state_lock:
                    self._view["last_input_retained"] = record["last_input_retained"]
            if interrupted or record["status"] == "cancelled":
                raise KeyboardInterrupt("本轮已取消")
            if callback_error is not None:
                raise callback_error
            if record["status"] != "completed":
                raise ApplicationRequestError(record)
            return _result_value(record["result"])
        finally:
            task.unsubscribe(queue)

    def _mutate(self, operation: str, payload: dict, request_id: str | None = None, on_event=None):
        """串行提交本入口的一次固定会话操作。

        作用:
            /new 完成之前不能从同一客户端把后续文字绑定到旧会话。
        参数:
            operation: 白名单操作名称。
            payload: 操作输入。
            request_id: 可选幂等标识。
            on_event: 可选调用线程展示回调。
        返回:
            已完成结果。
        行为说明:
            1. 同入口调用按顺序完成，其他入口仍可独立提交。
        """
        with self._operation_lock:
            self.last_result = None
            return self._wait(self._service._submit(self, operation, payload, request_id), on_event)

    def respond(
        self,
        prompt: str,
        *,
        images: Sequence[ImageInput] = (),
        image_labels: Sequence[str] = (),
        on_event=None,
        request_id: str | None = None,
    ) -> AgentTurnResult:
        """提交一次文字或图文输入。

        作用:
            保留 CLI 的 respond 接口，同时支持微信提供稳定请求 ID。
        参数:
            prompt: 原始输入文字。
            images: 已固定内容的图片快照。
            image_labels: 图片标签。
            on_event: 只在调用线程执行的展示回调。
            request_id: 可选稳定 ID；重复内容也不能跨来源或会话复用。
        返回:
            不含原始历史的 AgentTurnResult 展示投影。
        行为说明:
            1. 入口预检只用缓存，worker 执行前继续按实际会话校验。
        """
        images = tuple(images)
        if any(not isinstance(image, ImageInput) for image in images):
            raise ImageInputError("图片必须先经过加载和验证。")
        payload = {"prompt": prompt, "images": [asdict(image) for image in images], "image_labels": list(image_labels)}
        result = self._mutate("respond", payload, request_id, on_event)
        self.last_result = result
        return result

    def new_session(
        self, title: str | None = None, system_prompt: str | None = None, *, request_id: str | None = None
    ) -> SessionInfo:
        """从本入口会话配置新建并切换。

        作用:
            保留旧历史，避免继承另一入口当前配置。
        参数:
            title: 可选标题。
            system_prompt: 可选新提示词。
            request_id: 可选稳定操作标识。
        返回:
            新会话概要。
        行为说明:
            1. 变更通过 worker 队列和持久幂等记录完成。
        """
        return self._mutate("new_session", {"title": title, "system_prompt": system_prompt}, request_id)

    def reset(self, system_prompt: str | None = None, *, request_id: str | None = None) -> None:
        """重置本入口为新会话。

        作用:
            兼容 CLI /reset 与 /system，不删除旧记录。
        参数:
            system_prompt: 可选新提示词。
            request_id: 可选幂等标识。
        返回:
            None。
        行为说明:
            1. 只更新当前入口选择。
        """
        self._mutate("reset", {"system_prompt": system_prompt}, request_id)

    def resume_session(self, reference: str, *, request_id: str | None = None) -> SessionInfo:
        """选择一个明确的已有会话。

        作用:
            保留唯一前缀校验及恢复提示，不改变其他入口绑定。
        参数:
            reference: 完整 ID 或唯一前缀。
            request_id: 可选幂等标识。
        返回:
            目标会话概要。
        行为说明:
            1. worker 负责恢复，不自动重跑历史任务。
        """
        return self._mutate("resume_session", {"reference": reference}, request_id)

    def rename_session(self, title: str, *, request_id: str | None = None) -> SessionInfo:
        """修改本入口所选会话标题。

        作用:
            兼容已有会话命名能力。
        参数:
            title: 新标题。
            request_id: 可选幂等标识。
        返回:
            更新后概要。
        行为说明:
            1. 同会话其他入口共享标题，但不改变会话选择。
        """
        return self._mutate("rename_session", {"title": title}, request_id)

    def set_thinking(self, enabled: bool, *, request_id: str | None = None) -> None:
        """修改所选会话的思考设置。

        作用:
            沿用 Runtime 对模型能力的实际校验。
        参数:
            enabled: 是否启用。
            request_id: 可选幂等标识。
        返回:
            None。
        行为说明:
            1. 由 worker 修改并刷新绑定同会话的入口快照。
        """
        self._mutate("set_thinking", {"enabled": enabled}, request_id)

    def list_sessions(self) -> list[SessionInfo]:
        """读取可见会话列表。

        作用:
            支持 CLI 列表，排除私有默认模板。
        返回:
            会话概要列表。
        行为说明:
            1. SQL 在 worker 内执行。
        """
        return self._service._call(self, "list_sessions")

    def session_usage(self) -> SessionUsage | None:
        """读取本入口会话的最新持久统计。

        作用:
            保留启动、切换和失败后的用量展示。
        返回:
            SessionUsage 或未知值 None。
        行为说明:
            1. 查询经队列执行并同步缓存。
            2. 服务正在收尾时使用已发布快照，避免取消后的 CLI 统计刷新再次提交任务。
        """
        if not self._service._accepting:
            return _summary(self._field("session_usage"))
        return self._service._call(self, "session_usage")

    def session_details(self) -> dict:
        """查询本入口会话的完整关联审计。

        作用:
            保留 /history --ids 的显式查询能力。
        返回:
            已由 Runtime 隐藏图片字节的审计副本。
        行为说明:
            1. 不作为普通 respond 结果自动返回。
        """
        return self._service._call(self, "session_details")

    def history(self) -> list[dict]:
        """查询本入口会话的上下文展示副本。

        作用:
            保留 /history 能力，不暴露其他入口当前上下文。
        返回:
            图片已经摘要化的消息列表。
        行为说明:
            1. 后台先选择固定 Session，再读取 Runtime 历史。
        """
        return self._service._call(self, "history")

    def check_image_budget(self, images: Sequence[ImageInput]) -> None:
        """使用本入口快照执行非阻塞图片预检。

        作用:
            终端编辑回调不会等待微信长任务；真正提交仍由 worker 再验证。
        参数:
            images: 本次新增图片。
        返回:
            None。
        异常:
            ImageInputError: 附件类型、数量或累计体积不符合现有策略。
        行为说明:
            1. 预检不意味着执行时预算不会被共享会话的另一个入口改变。
        """
        count, total = self._field("image_budget", [0, 0])
        for image in images:
            if not isinstance(image, ImageInput):
                raise ImageInputError("图片必须先经过加载和验证。")
            count += 1
            total += len(image.data)
        if count > MAX_SESSION_IMAGES:
            raise ImageInputError(f"本段对话最多保留 {MAX_SESSION_IMAGES} 张图片，请 /reset 后开始新对话。")
        if total > MAX_SESSION_BASE64_BYTES:
            raise ImageInputError("本段对话的图片总量超过上限，请移除新附件或 /reset 后开始新对话。")

    def drain_memory_events(self) -> list[dict]:
        """按独立游标读取本入口后台记忆通知。

        作用:
            不直接消费 Runtime 队列，不影响其他客户端通知。
        返回:
            受控状态事件列表。
        行为说明:
            1. 短请求库连接在当前调用线程内关闭。
        """
        with self._state_lock:
            events, cursor = self._service.store.notifications(self.source, self.session_id, self._notification_cursor)
            self._notification_cursor = cursor
            return events

    def wait_for_memory(self, timeout: float | None = None) -> bool:
        """显式等待后台提取完成。

        作用:
            供隔离验收或调用方需要确认保存时使用，不关闭 Runtime。
        参数:
            timeout: 最长等待时间。
        返回:
            是否完成。
        行为说明:
            1. 等待发生在 Runtime owner 线程。
        """
        return self._service._call(self, "wait_for_memory", {"timeout": timeout})

    def lookup_request(self, request_id: str) -> dict | None:
        """查询当前来源的原请求状态与结果。

        作用:
            微信恢复使用精确 ID 继续取结果，不重新执行未知任务。
        参数:
            request_id: 已保存的应用请求标识。
        返回:
            状态、Session/Turn、结果对象与安全错误；不存在时为 None。
        异常:
            ValueError: 请求属于另一来源。
        行为说明:
            1. 不改变当前客户端选择，也不重跑模型。
        """
        record = self._service.store.get(request_id)
        if record is None:
            return None
        if record["source"] != self.source:
            raise ValueError("请求属于另一入口来源")
        keys = (
            "request_id",
            "source",
            "operation",
            "session_id",
            "turn_id",
            "status",
            "error_code",
            "error_message",
            "last_input_retained",
        )
        return {**{key: record[key] for key in keys}, "result": _result_value(record["result"])}

    def close(self) -> None:
        """分离当前客户端，保留服务与已经接受的请求。

        作用:
            终端退出不影响微信、Runtime 或后台记忆。
        返回:
            None。
        行为说明:
            1. 不调用 Runtime.close，不取消正在运行的任务。
        """
        with self._service._lock:
            self._closed = True
            self._service._clients.pop(self._identity, None)
