"""用真实临时 Runtime 验证统一应用的线程、会话、取消和恢复边界。"""

import copy
import json
import sqlite3
from concurrent.futures import Future
from threading import Event, Thread, get_ident
from types import SimpleNamespace

import pytest
from PIL import Image

from kairos import application as application_module
from kairos.application import ApplicationRequestError, ApplicationService
from kairos.application_store import ApplicationStore
from kairos.cancellation import CancellationToken, raise_if_cancelled
from kairos.legacy_images import load_image
from kairos.providers import OpenAICompatibleModel
from kairos.runtime import DEFAULT_SYSTEM_PROMPT, create_runtime


class Model:
    """记录实际请求，可返回脚本答案或暂停一个真实 Runtime 回合。"""

    model = "application-offline"

    def __init__(self, responses=None, *, entered=None, release=None):
        """准备完全离线的模型。

        作用:
            只替换上游模型，保留实际 Runtime、工具与 SQLite。
        参数:
            responses: 可选脚本响应序列。
            entered: 接收到暂停输入时设置的事件。
            release: 允许暂停输入继续的事件。
        返回:
            None。
        行为说明:
            1. 不创建网络客户端。
        """
        self.responses = iter(responses) if responses is not None else None
        self.requests = []
        self.threads = []
        self.entered, self.release = entered, release

    def complete(self, *, messages, tools):
        """记录请求并返回预设结果。

        作用:
            给真实 Agent 循环提供可预测模型行为。
        参数:
            messages: Runtime 实际组装的历史。
            tools: 实际工具定义。
        返回:
            统一模型响应。
        行为说明:
            1. hold 输入等待测试事件，其余直接返回。
        """
        self.requests.append(copy.deepcopy(messages))
        self.threads.append(get_ident())
        text = next(item["content"] for item in reversed(messages) if item["role"] == "user")
        if text == "hold" and self.entered is not None:
            self.entered.set()
            assert self.release.wait(5)
        if self.responses is not None:
            return next(self.responses)
        return {
            "content": f"回答：{text}" if isinstance(text, str) else "看到了图片",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cached_input_tokens": 0, "reasoning_tokens": 0},
        }

    def stream(self, *, messages, tools, on_text):
        """通过真实回调边界产生正文。

        作用:
            检查应用层将 UI 回调移出 Runtime 工作线程。
        参数:
            messages: 实际历史。
            tools: 实际工具定义。
            on_text: Agent 循环提供的同步正文回调。
        返回:
            完整模型结果。
        行为说明:
            1. 只回调正文；工具请求仍由原循环处理。
        """
        result = self.complete(messages=messages, tools=tools)
        if result.get("content"):
            on_text(result["content"])
        return result


def make_service(tmp_path, model, *, memory=False, extraction=None):
    """组装有线程记录的真实临时运行时服务。

    作用:
        避免测试把入口代理错误地当成 Runtime 本身。
    参数:
        tmp_path: 临时目录。
        model: 离线模型。
        memory: 是否启用临时记忆。
        extraction: 可选独立提取模型工厂。
    返回:
        服务、创建/关闭线程记录与 Runtime 引用容器。
    行为说明:
        1. 工厂及关闭均记录线程，业务数据只写临时目录。
    """
    life, runtimes = [], []

    def factory():
        """在应用工作线程构造实际 Runtime。

        作用:
            为测试保留真实数据库线程检查和执行链。
        返回:
            新 Runtime。
        行为说明:
            1. 仅使用显式临时路径和离线模型。
        """
        life.append(("create", get_ident()))
        runtime = create_runtime(
            model=model,
            db_path=tmp_path / "business.sqlite3",
            memory_dir=tmp_path / "memory",
            memory_enabled=memory,
            extraction_model_factory=extraction,
        )
        close = runtime.close

        def tracked_close():
            """记录真实 Runtime 关闭线程。

            作用:
                核对创建、使用和关闭的线程归属。
            返回:
                None。
            行为说明:
                1. 委托真实关闭，不伪造收尾成功。
            """
            life.append(("close", get_ident()))
            close()

        runtime.close = tracked_close
        runtimes.append(runtime)
        return runtime

    return ApplicationService(factory, request_path=tmp_path / "requests.sqlite3"), life, runtimes


def threaded(operation):
    """在独立入口线程运行一个可观察操作。

    作用:
        并发测试通过 Future 收集结果及 KeyboardInterrupt。
    参数:
        operation: 待执行函数。
    返回:
        线程和 Future。
    行为说明:
        1. 异常只写入 Future，不产生未处理线程警告。
    """
    result = Future()

    def run():
        """执行一次入口调用并捕获全部结果。

        作用:
            使测试可在主线程验证取消结果。
        返回:
            None。
        行为说明:
            1. BaseException 同样作为结果保留。
        """
        try:
            result.set_result(operation())
        except BaseException as error:
            result.set_exception(error)

    thread = Thread(target=run)
    thread.start()
    return thread, result


def test_clients_keep_a_b_a_history_usage_and_one_runtime_thread(tmp_path):
    """验证多个入口交替使用唯一 Runtime 而不串会话。

    作用:
        验证核心实际历史、累计用量及连接线程归属。
    参数:
        tmp_path: 隔离目录。
    返回:
        None。
    行为说明:
        1. A/B/A 三次输入分别取正确上下文，任一客户端关闭后其他入口继续。
    """
    model = Model()
    service, life, _ = make_service(tmp_path, model)
    try:
        assert life == []
        a = service.client("terminal:a")
        b = service.client("weixin:bot:owner")
        assert a.session_id != b.session_id
        a.respond("A1")
        b.respond("B1")
        result = a.respond("A2")
        assert result.session_id == a.session_id
        assert "A1" in str(model.requests[-1]) and "A2" in str(model.requests[-1])
        assert "B1" not in str(model.requests[-1])
        assert a.session_usage().usage.input_tokens == 20
        assert b.session_usage().usage.input_tokens == 10
        a.close()
        b.respond("B2")
        assert not any(kind == "close" for kind, _ in life)
        assert len({thread for _, thread in life} | set(model.threads)) == 1
        assert model.threads[0] != get_ident()
    finally:
        service.close()
    assert [kind for kind, _ in life] == ["create", "close"]
    assert life[0][1] == life[-1][1]


def test_new_clients_use_initial_defaults_and_new_inherits_own_session(tmp_path):
    """验证新入口用初始配置，入口新建沿用自己的配置。

    作用:
        避免微信最后活动状态污染终端的新会话。
    参数:
        tmp_path: 临时目录。
    返回:
        None。
    行为说明:
        1. 使用真实 DeepSeek 适配器能力判断，但不会请求网络。
    """
    model = OpenAICompatibleModel(model="deepseek-offline", client=SimpleNamespace())
    service, _, _ = make_service(tmp_path, model)
    try:
        a = service.client("a")
        a.reset(system_prompt="A 独立系统规则")
        a.set_thinking(True)
        b = service.client("b")
        assert b.system_prompt == DEFAULT_SYSTEM_PROMPT and b.thinking_enabled is False
        b.new_session(title="B 命名会话")
        assert b.system_prompt == DEFAULT_SYSTEM_PROMPT and b.thinking_enabled is False
        a.new_session(title="A 命名会话")
        assert a.system_prompt == "A 独立系统规则" and a.thinking_enabled is True
        c = service.client("c")
        assert c.system_prompt == DEFAULT_SYSTEM_PROMPT and c.thinking_enabled is False
        latest = service.client("latest", resume_latest=True)
        assert latest.session_id == a.session_id
        assert not ({session.session_id for session in a.list_sessions()} & service.store.templates())
        with pytest.raises(ApplicationRequestError):
            a.resume_session(next(iter(service.store.templates())))
    finally:
        service.close()


def test_events_run_in_caller_and_results_hide_raw_history(tmp_path):
    """验证事件线程和结果投影不包含原始模型载荷。

    作用:
        保留流式终端能力，不把 UI 回调放进 Runtime worker。
    参数:
        tmp_path: 临时目录。
    返回:
        None。
    行为说明:
        1. 在入口线程记录事件，检查持久结果没有原始消息或系统提示词。
    """
    model = Model()
    service, _, _ = make_service(tmp_path, model)
    try:
        client = service.client("terminal")
        client.reset(system_prompt="PRIVATE-SYSTEM-MARKER")
        events, threads = [], []

        def on_event(event):
            """记录展示回调所在调用线程。

            作用:
                验证回调不会直接从 Runtime worker 执行。
            参数:
                event: 已裁剪事件。
            返回:
                None。
            行为说明:
                1. 不调用模型或变更请求。
            """
            events.append(event)
            threads.append(get_ident())

        result = client.respond("正文", on_event=on_event, request_id="stable")
        assert set(threads) == {get_ident()}
        assert get_ident() not in model.threads
        assert any(event["event"] == "text_delta" for event in events)
        assert all("messages" not in event and "response" not in event and "tools" not in event for event in events)
        assert result.messages == [] and result.events == []
        saved = service.store.get("stable")["result"]
        assert "messages" not in saved["value"] and "events" not in saved["value"]
        assert "PRIVATE-SYSTEM-MARKER" not in json.dumps(saved["value"])
    finally:
        service.close()


def test_same_request_never_reexecutes_real_todo_and_conflicts_are_rejected(tmp_path):
    """验证稳定请求重复查询不会重复产生真实工具副作用。

    作用:
        检查来源、会话和载荷都进入应用幂等边界。
    参数:
        tmp_path: 临时业务与请求库目录。
    返回:
        None。
    行为说明:
        1. 实际执行一次待办写入，再重放相同 ID 并测试冲突。
    """
    model = Model(
        [
            {"tool_calls": [{"id": "add-one", "function": {"name": "add_todo", "arguments": '{"title":"只写一次"}'}}]},
            {"content": "已新增一次", "finish_reason": "stop"},
        ]
    )
    service, _, _ = make_service(tmp_path, model)
    try:
        a = service.client("a")
        first = a.respond("新增待办", request_id="one-write")
        again = a.respond("新增待办", request_id="one-write")
        assert first.turn_id == again.turn_id and len(model.requests) == 2
        with sqlite3.connect(tmp_path / "business.sqlite3") as connection:
            assert connection.execute("SELECT count(*) FROM todos").fetchone()[0] == 1
        with pytest.raises(ValueError):
            a.respond("改成另一个任务", request_id="one-write")
        with pytest.raises(ValueError):
            a.respond("不能把空标识当作新任务", request_id="")
        b = service.client("b", session_id=a.session_id)
        with pytest.raises(ValueError):
            b.respond("新增待办", request_id="one-write")
        with pytest.raises(ValueError):
            b.lookup_request("one-write")
        assert a.lookup_request("one-write")["result"].turn_id == first.turn_id
        a.new_session()
        with pytest.raises(ValueError):
            a.respond("新增待办", request_id="one-write")
    finally:
        service.close()


def test_image_snapshot_and_replayed_request_do_not_duplicate_images(tmp_path):
    """验证图片经过代理后保留快照，重复请求不重复加入历史。

    作用:
        保留原图片输入能力及请求幂等。
    参数:
        tmp_path: 合成图片与临时库目录。
    返回:
        None。
    行为说明:
        1. 删除原图片后提交内存快照，再用相同 ID 查询原结果。
    """
    image_path = tmp_path / "synthetic.png"
    Image.new("RGB", (8, 8), "red").save(image_path)
    image = load_image(image_path)
    image_path.unlink()
    model = Model()
    service, _, _ = make_service(tmp_path, model)
    try:
        client = service.client("images")
        first = client.respond("看图", images=(image,), image_labels=("图片 9",), request_id="image")
        again = client.respond("看图", images=(image,), image_labels=("图片 9",), request_id="image")
        assert first.turn_id == again.turn_id and len(model.requests) == 1
        assert image.data in json.dumps(model.requests)
        assert image.data not in json.dumps(client.history())
        assert client._field("image_budget")[0] == 1
    finally:
        service.close()


def test_ctrl_c_cancels_only_active_request_and_waits_for_worker(tmp_path, monkeypatch):
    """验证调用线程的 Ctrl+C 转为合作取消且不污染其他入口。

    作用:
        请求实际退出后才把 KeyboardInterrupt 交回 UI。
    参数:
        tmp_path: 隔离目录。
        monkeypatch: 注入取消已设置事件。
    返回:
        None。
    行为说明:
        1. 模型等待真实取消标记，后续另一个客户端可以正常执行。
    """
    cancellation_set, execution_stopped = Event(), Event()
    cancel = CancellationToken.cancel

    def observe_cancel(token):
        """在原取消操作之后通知离线模型。

        作用:
            不用时间猜测取消与模型检查的顺序。
        参数:
            token: 当前请求标记。
        返回:
            None。
        行为说明:
            1. 保留真实 Event 取消机制。
        """
        cancel(token)
        cancellation_set.set()

    monkeypatch.setattr(CancellationToken, "cancel", observe_cancel)

    class CancelModel(Model):
        def stream(self, *, messages, tools, on_text):
            """等待调用线程通过实际取消接口停止本轮。

            作用:
                区分 UI 已请求取消和执行已经停止。
            参数:
                messages: 真实历史。
                tools: 工具定义。
                on_text: 原 Agent 流回调。
            返回:
                普通输入返回正常结果。
            行为说明:
                1. cancel 输入等待标记；finally 记录执行停止。
            """
            if messages[-1]["content"] != "cancel":
                return super().stream(messages=messages, tools=tools, on_text=on_text)
            try:
                on_text("已开始")
                assert cancellation_set.wait(3)
                raise_if_cancelled()
            finally:
                execution_stopped.set()

    service, _, _ = make_service(tmp_path, CancelModel())
    try:
        a, b = service.client("a"), service.client("b")

        def interrupt(event):
            """在入口收到正文时模拟 Ctrl+C。

            作用:
                触发 RuntimeClient 的调用线程取消路径。
            参数:
                event: 展示事件。
            返回:
                非正文事件无返回值。
            异常:
                KeyboardInterrupt: 收到正文时模拟用户中断。
            行为说明:
                1. 不从外部强行终止 worker。
            """
            if event["event"] == "text_delta":
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            a.respond("cancel", request_id="cancelled", on_event=interrupt)
        assert execution_stopped.is_set()
        assert a.lookup_request("cancelled")["status"] == "cancelled"
        assert a.last_input_retained is False
        assert b.respond("正常输入").content == "回答：正常输入"
    finally:
        service.close()


def test_failed_session_selection_does_not_install_other_client_snapshot(tmp_path, monkeypatch):
    """验证选 A 失败时不把当前 B 的快照写给 A。

    作用:
        覆盖状态选择失败而 worker 仍停留另一会话的错误路径。
    参数:
        tmp_path: 临时目录。
        monkeypatch: 注入一次明确会话查询失败。
    返回:
        None。
    行为说明:
        1. B 已成功保留输入，A 的失败仍保持 A 身份且不继承 B 保留状态。
    """
    service, _, runtimes = make_service(tmp_path, Model())
    try:
        a = service.client("a")
        a.reset(system_prompt="A 的规则")
        b = service.client("b")
        b.respond("B 的内容")
        original = runtimes[0].session_store.get_session
        a_id = a.session_id

        def fail_a(reference):
            """只拒绝指定目标 A。

            作用:
                保持当前 B 可读，复现失败快照串话窗口。
            参数:
                reference: 目标会话。
            返回:
                非 A 的真实查询结果。
            异常:
                KeyError: 目标为 A 时故意拒绝。
            行为说明:
                1. 不更改原数据库。
            """
            if reference == a_id:
                raise KeyError("模拟 A 不可用")
            return original(reference)

        monkeypatch.setattr(runtimes[0].session_store, "get_session", fail_a)
        with pytest.raises(ApplicationRequestError):
            a.respond("A 后续", request_id="failed-a")
        assert a.session_id == a_id and a.system_prompt == "A 的规则"
        assert a.last_input_retained is False
        assert a.lookup_request("failed-a")["result"] is None
        assert b.last_input_retained is True
    finally:
        service.close()


def test_fatal_notification_failure_settles_queued_waiters(tmp_path, monkeypatch):
    """验证 worker 外层故障不留下永久等待的入口。

    作用:
        检查通知或存储故障时关闭接收并释放已排队 Future。
    参数:
        tmp_path: 临时目录。
        monkeypatch: 注入收尾通知故障。
    返回:
        None。
    行为说明:
        1. A 正在执行、B 已排队时使通知收集失败，B 必须收到终止错误。
    """
    entered, release, submitted = Event(), Event(), Event()
    service, life, _ = make_service(tmp_path, Model(entered=entered, release=release))
    a, b = service.client("a"), service.client("b")
    submit = service.store.submit

    def record_submit(*args, **kwargs):
        """在 B 真实持久接收后通知测试。

        作用:
            不靠固定等待猜测并发顺序。
        参数:
            args、kwargs: 原提交参数。
        返回:
            原提交结果。
        行为说明:
            1. 不改变请求状态。
        """
        result = submit(*args, **kwargs)
        if args[1] == "b":
            submitted.set()
        return result

    def fail_collection():
        """模拟统一通知存储失效。

        作用:
            验证外层 fatal 处理。
        返回:
            无。
        异常:
            OSError: 始终模拟本地存储失败。
        行为说明:
            1. 不访问外部资源。
        """
        raise OSError("模拟通知保存失败")

    monkeypatch.setattr(service.store, "submit", record_submit)
    first_thread, first = threaded(lambda: a.respond("hold"))
    assert entered.wait(3)
    second_thread, second = threaded(lambda: b.respond("B 等待"))
    assert submitted.wait(3)
    monkeypatch.setattr(service, "_collect_memory", fail_collection)
    release.set()
    try:
        assert first.result(3).content == "回答：hold"
        with pytest.raises(RuntimeError):
            second.result(3)
        with pytest.raises(RuntimeError):
            service.client("new")
        with pytest.raises(OSError):
            service.close()
        assert life[-1][0] == "close"
    finally:
        release.set()
        first_thread.join(3)
        second_thread.join(3)
        if service._worker.is_alive():
            with pytest.raises(OSError):
                service.close()


def test_constructor_failure_waits_for_runtime_close(tmp_path):
    """验证启动恢复失败时构造函数等待 worker 真正关闭。

    作用:
        防止入口提前释放全局锁时后台仍在关闭业务资源。
    参数:
        tmp_path: 临时目录。
    返回:
        None。
    行为说明:
        1. 用阻塞关闭模拟收尾，构造异常只能在释放收尾后返回。
    """
    path = tmp_path / "requests.sqlite3"
    store = ApplicationStore(path)
    store.submit("recovery", "a", "respond", "missing", {"prompt": "test", "images": [], "image_labels": []})
    store.start("recovery")
    store.link_turn("recovery", {"session_id": "missing", "turn_id": "turn", "user_message_id": "message"})
    closing, release, closed = Event(), Event(), Event()

    def factory():
        """创建可控恢复故障的真实 Runtime。

        作用:
            保留数据库释放行为，只注入读取失败和关闭屏障。
        返回:
            实际 Runtime。
        行为说明:
            1. 全部路径均在临时目录。
        """
        runtime = create_runtime(model=Model(), db_path=tmp_path / "business.sqlite3", memory_enabled=False)
        close = runtime.close

        def fail_inspect(_):
            """模拟恢复读取失败。

            作用:
                让启动进入失败收尾。
            参数:
                _: 未使用的会话 ID。
            返回:
                无。
            异常:
                RuntimeError: 始终触发模拟故障。
            行为说明:
                1. 不变更记录。
            """
            raise RuntimeError("模拟恢复失败")

        def wait_close():
            """阻塞收尾直到测试明确允许。

            作用:
                验证构造函数并未只等待 ready 异常。
            返回:
                None。
            行为说明:
                1. 最终执行真实数据库关闭。
            """
            closing.set()
            assert release.wait(3)
            close()
            closed.set()

        runtime.session_store.inspect = fail_inspect
        runtime.close = wait_close
        return runtime

    thread, construction = threaded(lambda: ApplicationService(factory, request_path=path))
    try:
        assert closing.wait(3)
        assert not construction.done()
        release.set()
        with pytest.raises(RuntimeError, match="模拟恢复失败"):
            construction.result(3)
        assert closed.is_set()
    finally:
        release.set()
        thread.join(3)


def test_restart_recovers_exact_completed_turn_without_model_replay(tmp_path, monkeypatch):
    """验证 Turn 已完成而请求结果尚未保存时精确恢复。

    作用:
        覆盖跨两份 SQLite 提交的执行结果窗口。
    参数:
        tmp_path: 临时目录。
        monkeypatch: 注入一次应用结果保存失败。
    返回:
        None。
    行为说明:
        1. 恢复只读原 Turn，离线模型的请求次数保持不变。
    """
    model = Model()
    service, _, _ = make_service(tmp_path, model)
    client = service.client("a")
    session_id = client.session_id
    finish = service.store.finish

    def fail_completed(request_id, status, **kwargs):
        """只阻止应用结果提交。

        作用:
            保留已经完成的真实 Runtime Turn。
        参数:
            request_id、status、kwargs: 原保存参数。
        返回:
            非完成路径的原保存结果。
        异常:
            OSError: completed 状态模拟提交失败。
        行为说明:
            1. 不回滚业务库已完成 Turn。
        """
        if status == "completed":
            raise OSError("模拟结果保存失败")
        return finish(request_id, status, **kwargs)

    monkeypatch.setattr(service.store, "finish", fail_completed)
    with pytest.raises(OSError):
        client.respond("精确恢复的答案", request_id="recover-result")
    assert client.last_input_retained is True
    original = service.store.get("recover-result")
    assert original["status"] == "running" and original["turn_id"]
    service.close()
    service, _, _ = make_service(tmp_path, model)
    try:
        restored = service.client("a", session_id=session_id)
        state = restored.lookup_request("recover-result")
        assert state["status"] == "completed" and state["turn_id"] == original["turn_id"]
        assert state["result"].content == "回答：精确恢复的答案"
        assert restored.respond("精确恢复的答案", request_id="recover-result").turn_id == original["turn_id"]
        assert len(model.requests) == 1
    finally:
        service.close()


def test_queued_ctrl_c_returns_before_other_entry_finishes(tmp_path, monkeypatch):
    """验证排队终端取消不等待或取消微信的当前回合。

    作用:
        覆盖 queued 原子取消与 running 合作取消的区别。
    参数:
        tmp_path: 临时目录。
        monkeypatch: 在终端等待队列处注入一次 Ctrl+C。
    返回:
        None。
    行为说明:
        1. 微信模型被事件阻塞时，终端取消必须已经返回。
    """
    entered, release = Event(), Event()
    service, _, _ = make_service(tmp_path, Model(entered=entered, release=release))
    wx, terminal = service.client("weixin"), service.client("terminal")
    original_subscribe = application_module._Task.subscribe

    def subscribe_with_interrupt(task):
        """仅向终端请求的调用线程注入中断。

        作用:
            不向 worker 发送进程信号，不影响微信等待队列。
        参数:
            task: 正在建立订阅的请求。
        返回:
            原队列，仅终端首次 get 抛中断。
        行为说明:
            1. 委托真实订阅，以便正常移除监听者。
        """
        queue = original_subscribe(task)
        if task.record["source"] == "terminal":
            get = queue.get
            interrupted = False

            def once(*args, **kwargs):
                """模拟终端在等待任务时按 Ctrl+C。

                作用:
                    驱动 RuntimeClient 的真实等待异常路径。
                参数:
                    args、kwargs: Queue.get 参数。
                返回:
                    后续委托队列读取。
                异常:
                    KeyboardInterrupt: 首次等待时抛出。
                行为说明:
                    1. 只影响当前任务队列。
                """
                nonlocal interrupted
                if not interrupted:
                    interrupted = True
                    raise KeyboardInterrupt
                return get(*args, **kwargs)

            queue.get = once
        return queue

    monkeypatch.setattr(application_module._Task, "subscribe", subscribe_with_interrupt)
    wx_thread, wx_result = threaded(lambda: wx.respond("hold", request_id="wx-long"))
    try:
        assert entered.wait(3)
        terminal_thread, terminal_result = threaded(lambda: terminal.respond("不应执行", request_id="terminal-queued"))
        with pytest.raises(KeyboardInterrupt):
            terminal_result.result(2)
        terminal_thread.join(2)
        assert not wx_result.done()
        assert service._tasks["wx-long"].token.is_cancelled is False
        assert terminal.lookup_request("terminal-queued")["status"] == "cancelled"
        release.set()
        assert wx_result.result(3).content == "回答：hold"
        assert service._worker.is_alive()
    finally:
        release.set()
        wx_thread.join(3)
        service.close()


def test_late_result_does_not_replace_newer_shared_session_snapshot(tmp_path):
    """验证慢 UI 消费旧结果不会覆盖另一入口的新标题。

    作用:
        保持会话共享字段与本请求私有完成事实的不同更新顺序。
    参数:
        tmp_path: 隔离目录。
    返回:
        None。
    行为说明:
        1. A 暂停消费正文，B 在同会话重命名，再让 A 接收旧结果。
    """
    service, _, _ = make_service(tmp_path, Model())
    entered, release = Event(), Event()
    a = service.client("a")
    b = service.client("b", session_id=a.session_id)

    def slow_callback(event):
        """只暂停调用线程，不暂停模型执行。

        作用:
            确保已完成任务的结果可以晚于另一入口变更被消费。
        参数:
            event: 展示事件。
        返回:
            None。
        行为说明:
            1. 使用事件屏障而不是固定睡眠。
        """
        if event["event"] == "text_delta":
            entered.set()
            assert release.wait(3)

    thread, result = threaded(lambda: a.respond("hello", on_event=slow_callback))
    try:
        assert entered.wait(3)
        b.rename_session("fresh-title")
        assert a.session_info.title == "fresh-title"
        release.set()
        assert result.result(3).content == "回答：hello"
        assert a.session_info.title == b.session_info.title == "fresh-title"
        assert a.last_input_retained is True
    finally:
        release.set()
        thread.join(3)
        service.close()


def test_read_queries_preserve_failed_input_after_real_tool_execution(tmp_path):
    """验证错误后查询用量不会把已执行输入误判成可重发草稿。

    作用:
        覆盖 CLI 先刷新用量、再读取 last_input_retained 的正常流程。
    参数:
        tmp_path: 隔离目录。
    返回:
        None。
    行为说明:
        1. A 已执行真实待办后模型失败；B 切走，再查询 A 用量和历史。
    """

    class FailingModel(Model):
        def complete(self, **kwargs):
            """将脚本异常从模型边界抛出。

            作用:
                保留真实工具副作用并模拟后续模型请求失败。
            参数:
                kwargs: 原模型请求。
            返回:
                普通脚本结果。
            异常:
                脚本异常: 当前响应是 Exception 时抛出。
            行为说明:
                1. 不伪造工具执行状态。
            """
            result = super().complete(**kwargs)
            if isinstance(result, Exception):
                raise result
            return result

    model = FailingModel(
        [
            {"tool_calls": [{"id": "todo", "function": {"name": "add_todo", "arguments": '{"title":"保留副作用"}'}}]},
            RuntimeError("模拟第二次模型请求失败"),
            {"content": "B 正常"},
        ]
    )
    service, _, _ = make_service(tmp_path, model)
    try:
        a, b = service.client("a"), service.client("b")
        with pytest.raises(ApplicationRequestError):
            a.respond("新增", request_id="failed-after-tool")
        assert a.last_input_retained is True
        b.respond("B")
        a.session_usage()
        a.history()
        a.session_details()
        assert a.last_input_retained is True
        with sqlite3.connect(tmp_path / "business.sqlite3") as connection:
            assert connection.execute("SELECT count(*) FROM todos").fetchone()[0] == 1
    finally:
        service.close()


def test_control_request_replay_and_resume_notice(tmp_path):
    """验证控制操作幂等及显式恢复提示不被二次恢复清空。

    作用:
        保留 /new 和 /resume 的完整旧能力。
    参数:
        tmp_path: 临时目录。
    返回:
        None。
    行为说明:
        1. 先保存遗留 running Turn，再通过代理恢复并重放同一新建请求。
    """
    with create_runtime(model=Model(), db_path=tmp_path / "business.sqlite3", memory_enabled=False) as runtime:
        interrupted_session = runtime.session_id
        runtime.session_store.begin_turn(interrupted_session, {"role": "user", "content": "中断输入"})
    service, _, _ = make_service(tmp_path, Model())
    try:
        client = service.client("a")
        client.resume_session(interrupted_session)
        assert client.recovered_turns == 1
        assert client.session_details()["turns"][0]["status"] == "interrupted"
        assert client.recovered_turns == 1
        before = client.session_id
        created = client.new_session("只新建一次", request_id="new-stable")
        assert client.lookup_request("new-stable")["result"] == created
        client.resume_session(before)
        assert client.new_session("只新建一次", request_id="new-stable").session_id == created.session_id
        with sqlite3.connect(tmp_path / "business.sqlite3") as connection:
            assert (
                connection.execute("SELECT count(*) FROM sessions WHERE title = ?", ("只新建一次",)).fetchone()[0] == 1
            )
    finally:
        service.close()


def test_client_detach_keeps_memory_and_notifications_are_independent(tmp_path):
    """验证入口离开后记忆继续运行，通知各自读取。

    作用:
        不让客户端 close 关闭共享后台，也不让通知消费相互抢占。
    参数:
        tmp_path: 临时数据库与记忆目录。
    返回:
        None。
    行为说明:
        1. 后台提取被屏障暂停时关闭 A，再由同来源另两个客户端独立读取结果。
    """
    entered, release = Event(), Event()

    class ExtractModel(Model):
        def complete(self, **kwargs):
            """在第一次后台请求等待测试释放。

            作用:
                保证关闭入口时后台工作确实仍未完成。
            参数:
                kwargs: 真实提取上下文。
            返回:
                脚本提取响应。
            行为说明:
                1. 后续调用直接进入实际记忆工具循环。
            """
            if not self.requests:
                entered.set()
                assert release.wait(3)
            return super().complete(**kwargs)

    extraction = ExtractModel(
        [
            {
                "tool_calls": [
                    {
                        "id": "core",
                        "function": {"name": "write_core", "arguments": '{"content":"测试用户偏好简短中文交流。"}'},
                    }
                ]
            },
            {"content": '{"status":"completed"}'},
        ]
    )
    service, life, _ = make_service(tmp_path, Model(), memory=True, extraction=lambda: extraction)
    try:
        a = service.client("shared-source")
        b = service.client("shared-source", session_id=a.session_id)
        a.respond("我喜欢简短中文交流")
        assert entered.wait(3)
        a.close()
        assert not any(kind == "close" for kind, _ in life)
        release.set()
        assert b.wait_for_memory(3)
        events = b.drain_memory_events()
        assert len(events) == 1 and events[0]["status"] == "completed"
        assert events[0]["core_additions"] == 1
        assert b.drain_memory_events() == []
        c = service.client("shared-source", session_id=b.session_id)
        assert c.drain_memory_events() == events
        unrelated = service.client("other-source", session_id=b.session_id)
        assert unrelated.drain_memory_events() == []
    finally:
        release.set()
        service.close()


def test_stop_accepting_prevents_late_submit_but_keeps_lookup(tmp_path):
    """验证关停准入关闭后不会接收迟到入口任务。

    作用:
        覆盖停止微信循环与等待其退出之间的提交竞态。
    参数:
        tmp_path: 临时目录。
    返回:
        None。
    行为说明:
        1. 关闭准入不关闭 Runtime，已经完成的结果仍可查询。
    """
    service, _, _ = make_service(tmp_path, Model())
    try:
        client = service.client("a")
        client.respond("原请求", request_id="done")
        service.stop_accepting()
        with pytest.raises(RuntimeError):
            client.respond("迟到请求", request_id="late")
        assert client.session_usage().usage.input_tokens == 10
        assert client.lookup_request("done")["status"] == "completed"
        assert client.lookup_request("late") is None
        client.close()
    finally:
        service.close()
