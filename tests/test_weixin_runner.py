"""验证同进程微信接收、应用请求恢复与线程各自关闭存储。"""

import copy
import queue
import sqlite3
import threading
from contextlib import closing
from types import SimpleNamespace

import pytest

from kairos.channel_app import ChannelApp
from kairos.channels.base import InboundMessage, ReceiveBatch, SendReceipt
from kairos.channels.store import INTERRUPTED_REPLY, ChannelStore
from kairos.weixin_channel import WeixinChannelRunner


def incoming(message_id, text="问题"):
    """构造固定本人身份的合成微信输入。

    作用:
        为并发接收和恢复测试提供稳定来源。
    参数:
        message_id: 外部消息 ID。
        text: 输入文字或本地命令。
    返回:
        InboundMessage。
    行为说明:
        1. 不含真实账号或隐私数据。
    """
    return InboundMessage("weixin", "test-bot", "test-owner", "test-owner", message_id, text)


class FakeClient:
    """模拟应用客户端，仅记录操作与查询，不创建 Runtime。"""

    def __init__(self, *, source, session_id=None, requests=None, started=None, release=None, cancelled=False):
        """保存测试会话和可选执行屏障。

        作用:
            模拟后台忙时阻塞 respond，但不阻塞独立接收线程。
        参数:
            source: 稳定客户端来源。
            session_id: 初始会话。
            requests: 可查询的合成后台账本。
            started, release: 可选开始及放行事件。
            cancelled: 放行后是否抛取消异常。
        返回:
            无返回值。
        行为说明:
            1. 所有结果均在内存中构造。
        """
        self.source = source
        self.session_id = session_id or "first-session"
        self.requests = requests if requests is not None else {}
        self.started = started
        self.release = release
        self.cancelled = cancelled
        self.calls = []
        self.lookups = []
        self.closed = False

    @property
    def session_info(self):
        """返回当前会话的合成概要。

        作用:
            为 /session 提供客户端兼容视图。
        参数:
            无。
        返回:
            含 session_id 和 title 的对象。
        行为说明:
            1. 不访问业务数据库。
        """
        return SimpleNamespace(session_id=self.session_id, title="测试会话")

    def resume_session(self, session_id):
        """选择明确的既有会话。

        作用:
            模拟轻量客户端切换选中 Session。
        参数:
            session_id: 已绑定会话。
        返回:
            None。
        行为说明:
            1. 不产生模型调用。
        """
        self.session_id = session_id

    def respond(self, text, *, request_id):
        """阻塞或完成一次合成应用请求。

        作用:
            验证稳定 ID、顺序与取消后保留 processing。
        参数:
            text: 输入正文。
            request_id: 已落盘的稳定关联。
        返回:
            含 content 和 turn_id 的结果对象。
        异常:
            KeyboardInterrupt: 测试指定取消当前请求。
        行为说明:
            1. 取消也先记录调用，模拟任务可能已经执行部分工具。
        """
        self.calls.append(("respond", request_id, self.session_id, text))
        if self.started:
            self.started.set()
        if self.release:
            assert self.release.wait(5), "测试没有释放合成模型执行"
        if self.cancelled:
            raise KeyboardInterrupt
        result = SimpleNamespace(content=f"答案：{text}", turn_id=f"turn-{request_id}")
        self.requests[request_id] = {
            "request_id": request_id,
            "source": self.source,
            "operation": "respond",
            "status": "completed",
            "session_id": self.session_id,
            "result": result,
        }
        return result

    def new_session(self, *, title=None, request_id=None):
        """记录一次带稳定 ID 的会话创建。

        作用:
            检查 /new 对后续输入形成绑定屏障。
        参数:
            title: 可选标题。
            request_id: 已持久化控制操作 ID。
        返回:
            新会话概要。
        行为说明:
            1. 测试每次创建使用不同 ID，便于发现误重做。
        """
        self.calls.append(("new_session", request_id, self.session_id, title))
        self.session_id = f"new-session-{len(self.calls)}"
        return self.session_info

    def lookup_request(self, request_id):
        """只读查询合成账本。

        作用:
            保证恢复不会借查询名义再次执行任务。
        参数:
            request_id: 原稳定请求 ID。
        返回:
            状态字典或 None。
        行为说明:
            1. 单独记录查询，不添加 calls 中的执行操作。
        """
        self.lookups.append(request_id)
        return self.requests.get(request_id)

    def drain_memory_events(self):
        """返回空的本客户端通知队列。

        作用:
            兼容通道的受控记忆通知轮询。
        参数:
            无。
        返回:
            空列表。
        行为说明:
            1. 不访问其他客户端通知。
        """
        return []

    def close(self):
        """标记客户端已分离。

        作用:
            验证通道收尾不关闭真实后台 Runtime。
        参数:
            无。
        返回:
            None。
        行为说明:
            1. 只改变轻量客户端状态。
        """
        self.closed = True


class FakeAdapter:
    """让各线程分别持有实例，同时通过 Queue 共享合成外部平台。"""

    channel, account_id, owner_id = "weixin", "test-bot", "test-owner"

    def __init__(self, batches=None, sent=None):
        """绑定测试消息源。

        作用:
            不用共享真实 HTTPS 客户端也能观察接收与发送并行。
        参数:
            batches: 供接收线程消费的队列。
            sent: 保存发送结果的列表。
        返回:
            无返回值。
        行为说明:
            1. 记录构造线程，所有操作要求在该线程执行。
        """
        self.batches = batches or queue.Queue()
        self.sent = sent if sent is not None else []
        self.owner_thread = threading.get_ident()

    def poll(self, checkpoint):
        """有界等待一批合成平台消息。

        作用:
            用短等待模拟可退出的长轮询。
        参数:
            checkpoint: 上次持久接收位置。
        返回:
            有消息的新批次，或保留旧 checkpoint 的空批次。
        行为说明:
            1. 不同线程不能复用当前适配器。
        """
        assert threading.get_ident() == self.owner_thread
        try:
            batch = self.batches.get(timeout=0.05)
        except queue.Empty:
            return ReceiveBatch((), copy.deepcopy(checkpoint))
        return ReceiveBatch(tuple(batch), {"cursor": batch[-1].message_id})

    def send(self, target, text, client_id, checkpoint):
        """记录一次合成发送。

        作用:
            验证答案只由处理线程投递。
        参数:
            target, text, client_id, checkpoint: 已保存的发送参数。
        返回:
            明确业务成功的 SendReceipt。
        行为说明:
            1. 不执行模型或平台网络请求。
        """
        assert threading.get_ident() == self.owner_thread
        self.sent.append((target, text, client_id))
        return SendReceipt(True)


def test_receiver_persists_new_messages_while_application_request_is_busy(tmp_path):
    """验证后台请求阻塞期间微信接收仍能推进持久游标。

    作用:
        直接覆盖此次同进程多入口改造的主要行为。
    参数:
        tmp_path: 隔离 SQLite 目录。
    返回:
        None。
    行为说明:
        1. 用事件停住首个 respond，再收到第二条消息并检查其已经落盘。
        2. 每个连接在创建线程关闭，且所有工作线程退出后 join 才成功。
    """
    batches, sent = queue.Queue(), []
    started, release, second_saved = threading.Event(), threading.Event(), threading.Event()
    clients, adapters, stores = [], [], []

    class TrackingStore(ChannelStore):
        """记录数据库的创建与关闭线程。"""

        def __init__(self, path):
            """打开一个线程私有连接。

            作用:
                观察连接生命周期。
            参数:
                path: 临时数据库路径。
            返回:
                无返回值。
            行为说明:
                1. 底层仍使用真实 SQLite。
            """
            self.owner_thread = threading.get_ident()
            self.closed_by = None
            super().__init__(path)
            stores.append(self)

        def accept_batch(self, channel, account, batch):
            """实际保存输入后通知测试线程。

            作用:
                不以 poll 返回代替真正持久化完成。
            参数:
                channel, account, batch: 待保存的合法批次。
            返回:
                新增输入数。
            行为说明:
                1. 确认连接只在创建线程使用。
            """
            assert threading.get_ident() == self.owner_thread
            result = super().accept_batch(channel, account, batch)
            if any(item.message_id == "second" for item in batch.messages):
                second_saved.set()
            return result

        def close(self):
            """在创建线程释放连接。

            作用:
                检查主线程没有替后台关闭 SQLite。
            参数:
                无。
            返回:
                None。
            行为说明:
                1. 关闭前记录实际线程。
            """
            self.closed_by = threading.get_ident()
            assert self.closed_by == self.owner_thread
            super().close()

    def adapter_factory():
        """为每个微信线程创建独立适配器。

        作用:
            检查网络状态没有在线程间共享。
        参数:
            无。
        返回:
            新 FakeAdapter。
        行为说明:
            1. 只共享模拟平台队列。
        """
        result = FakeAdapter(batches, sent)
        adapters.append(result)
        return result

    def client_factory(**kwargs):
        """返回可以停住执行的假应用客户端。

        作用:
            观察接收不依赖模型执行进度。
        参数:
            kwargs: source 与 session_id。
        返回:
            FakeClient。
        行为说明:
            1. 不创建模型和业务数据库。
        """
        result = FakeClient(**kwargs, started=started, release=release)
        clients.append(result)
        return result

    runner = WeixinChannelRunner(
        tmp_path / "channels.sqlite3",
        adapter_factory,
        client_factory,
        max_messages=1,
        report=lambda _: None,
        store_factory=TrackingStore,
    )
    batches.put((incoming("first"),))
    try:
        runner.start()
        assert started.wait(3)
        batches.put((incoming("second"),))
        assert second_saved.wait(3)
        with closing(ChannelStore(tmp_path / "channels.sqlite3")) as check:
            assert check.checkpoint("weixin", "test-bot") == {"cursor": "second"}
            assert check.next_queued().message.message_id == "second"
            assert len(check.pending_requests("weixin", "test-bot")) == 1
        assert not sent and runner.running
        release.set()
        assert runner.join(3)
    finally:
        release.set()
        runner.stop()
        assert runner.join(5)
    assert runner.error is None and runner.processed == 1
    assert len(clients) == len(sent) == 1
    assert clients[0].closed
    assert len(adapters) == 2 and adapters[0] is not adapters[1]
    assert len({adapter.owner_thread for adapter in adapters}) == 2
    assert all(store.closed_by == store.owner_thread for store in stores)


@pytest.mark.parametrize("operation", ["respond", "new_session"])
def test_completed_application_request_is_recovered_without_execution(tmp_path, operation):
    """查询已完成请求并恢复答案或 /new 绑定，不重复操作。

    作用:
        覆盖后台完成但通道尚未保存结果时退出的窗口。
    参数:
        tmp_path: 隔离目录。
        operation: 普通回合或会话创建。
    返回:
        None。
    行为说明:
        1. 只查询原 ID，缓存结果落盘后才允许处理后续消息。
    """
    item = incoming("before-restart", "/new 新会话" if operation == "new_session" else "旧请求")
    source = ChannelApp.source_for(item)
    result = SimpleNamespace(content="后台已保存的答案", turn_id="original-turn", session_id="new-session")
    client = FakeClient(
        source=source,
        session_id="old-session",
        requests={
            "stable-request": {
                "request_id": "stable-request",
                "source": source,
                "operation": operation,
                "status": "completed",
                "session_id": "old-session",
                "result": result,
            },
        },
    )
    path = tmp_path / "channels.sqlite3"
    with closing(ChannelStore(path)) as original:
        original.accept_batch("weixin", "test-bot", ReceiveBatch((item,), {"cursor": "saved"}))
        original.bind(item, "old-session")
        record_id = original.next_queued().id
        original.claim(record_id, "old-session", request_id="stable-request")
    with closing(ChannelStore(path)) as store:
        app = ChannelApp(store, FakeAdapter(), lambda **kwargs: client, report=lambda _: None)
        try:
            assert app.process_next()
            record = store.get(record_id)
            assert record.status == "ready" and record.backend_request_id == "stable-request"
            assert client.calls == [] and client.lookups == ["stable-request"]
            if operation == "respond":
                assert record.response == result.content and record.turn_id == result.turn_id
            else:
                assert store.binding(item) == result.session_id
                assert record.session_id == result.session_id
        finally:
            app.close()


@pytest.mark.parametrize("status", [None, "running", "queued", "failed", "cancelled", "interrupted"])
def test_recovery_never_reexecutes_unknown_requests_or_passes_running_new_barrier(tmp_path, status):
    """等待运行中请求，其他未知终态保守处理，均不重跑。

    作用:
        防止连接器重启后 /new 越过尚未完成的先前输入。
    参数:
        tmp_path: 隔离目录。
        status: 后台返回状态或不存在。
    返回:
        None。
    行为说明:
        1. 后续 /new 仍为 queued；原模型工具没有再次调用。
    """
    first, switch = incoming("first"), incoming("switch", "/new")
    source = ChannelApp.source_for(first)
    requests = (
        {}
        if status is None
        else {
            "stable": {
                "request_id": "stable",
                "source": source,
                "operation": "respond",
                "status": status,
                "session_id": "old-session",
                "result": None,
            }
        }
    )
    client = FakeClient(source=source, session_id="old-session", requests=requests)
    with closing(ChannelStore(tmp_path / "channels.sqlite3")) as store:
        store.accept_batch("weixin", "test-bot", ReceiveBatch((first, switch), {}))
        store.bind(first, "old-session")
        record_id = store.next_queued().id
        store.claim(record_id, "old-session", request_id="stable")
        app = ChannelApp(store, FakeAdapter(), lambda **kwargs: client, report=lambda _: None)
        try:
            assert app.process_next() is (status not in {"running", "queued"})
            assert client.calls == []
            assert store.next_queued().message == switch
            assert store.binding(switch) == "old-session"
            if status not in {"running", "queued"}:
                assert store.get(record_id).response == INTERRUPTED_REPLY
        finally:
            app.close()


def test_new_binding_and_reply_are_committed_atomically(tmp_path):
    """在回复写入故障时回滚 /new 的绑定。

    作用:
        防止后续输入进入新会话但控制请求尚未可靠落盘。
    参数:
        tmp_path: 隔离 SQLite 目录。
    返回:
        None。
    行为说明:
        1. 用真实 SQLite 触发器注入故障，保持旧绑定与 processing 关联。
    """
    item = incoming("new", "/new")
    with closing(ChannelStore(tmp_path / "channels.sqlite3")) as store:
        store.accept_batch("weixin", "test-bot", ReceiveBatch((item,), {}))
        store.bind(item, "old-session")
        record_id = store.next_queued().id
        store.claim(record_id, "old-session", request_id="stable-new")
        store.connection.execute(
            """CREATE TRIGGER fail_ready BEFORE UPDATE ON channel_inbox
               WHEN NEW.status = 'ready' BEGIN SELECT RAISE(FAIL, 'synthetic failure'); END"""
        )
        with pytest.raises(sqlite3.IntegrityError):
            store.prepare_reply(record_id, "已创建", ["已创建"], kind="command", binding_session_id="new-session")
        assert store.binding(item) == "old-session"
        assert store.get(record_id).status == "processing"
        assert store.get(record_id).backend_request_id == "stable-new"


def test_recovery_rejects_result_associated_with_another_session(tmp_path):
    """拒绝将错误会话的已完成结果回给当前微信对话。

    作用:
        关联损坏时按原执行身份校验，/new 也应比较创建前的旧会话。
    参数:
        tmp_path: 隔离目录。
    返回:
        None。
    行为说明:
        1. 不回发异会话正文，不重新执行任务，只保存保守提示。
    """
    item = incoming("mismatched")
    source = ChannelApp.source_for(item)
    client = FakeClient(
        source=source,
        session_id="correct-session",
        requests={
            "stable": {
                "request_id": "stable",
                "source": source,
                "operation": "respond",
                "status": "completed",
                "session_id": "another-session",
                "result": SimpleNamespace(content="不能回发的其他会话内容", turn_id="other-turn"),
            },
        },
    )
    with closing(ChannelStore(tmp_path / "channels.sqlite3")) as store:
        store.accept_batch("weixin", "test-bot", ReceiveBatch((item,), {}))
        store.bind(item, "correct-session")
        record_id = store.next_queued().id
        store.claim(record_id, "correct-session", request_id="stable")
        app = ChannelApp(store, FakeAdapter(), lambda **kwargs: client, report=lambda _: None)
        try:
            assert app.process_next()
            assert store.get(record_id).response == INTERRUPTED_REPLY
            assert client.calls == []
        finally:
            app.close()


def test_stable_request_link_is_saved_before_execution_and_new_orders_followup(tmp_path):
    """检查执行前关联落盘及 A、/new、B 的会话归属。

    作用:
        输入已成批接收时也不能提前给 B 绑定旧会话。
    参数:
        tmp_path: 隔离目录。
    返回:
        None。
    行为说明:
        1. 模型和控制操作均取得非空请求 ID，绑定只在 /new 完成后切换。
    """
    items = (incoming("a", "A"), incoming("new", "/new"), incoming("b", "B"))
    client = FakeClient(source=ChannelApp.source_for(items[0]))
    with closing(ChannelStore(tmp_path / "channels.sqlite3")) as store:
        store.accept_batch("weixin", "test-bot", ReceiveBatch(items, {}))
        original_respond = client.respond

        def checked_respond(text, *, request_id):
            """在模拟后台执行前检查请求关联已提交。

            作用:
                不用事后状态推测写入顺序。
            参数:
                text, request_id: 本次应用请求。
            返回:
                原假客户端结果。
            行为说明:
                1. 通过第二个连接检查已提交状态。
            """
            with closing(ChannelStore(tmp_path / "channels.sqlite3")) as observer:
                pending = observer.pending_requests("weixin", "test-bot")
                assert len(pending) == 1 and pending[0].backend_request_id == request_id
            return original_respond(text, request_id=request_id)

        client.respond = checked_respond
        app = ChannelApp(store, FakeAdapter(), lambda **kwargs: client, report=lambda _: None)
        try:
            assert app.process_next() and app.process_next() and app.process_next()
            assert [call[0] for call in client.calls] == ["respond", "new_session", "respond"]
            assert client.calls[0][2] == "first-session"
            assert client.calls[2][2] == store.binding(items[2]) != "first-session"
            assert len({call[1] for call in client.calls}) == 3
            assert all(call[1] for call in client.calls)
        finally:
            app.close()


def test_stop_and_cancel_keep_request_for_recovery_without_thread_traceback(tmp_path, capsys):
    """取消处理中请求后安全退出线程并保留可查询关联。

    作用:
        应用关闭时不跨线程关库，也不因 KeyboardInterrupt 输出敏感 traceback。
    参数:
        tmp_path: 隔离目录。
        capsys: 捕获线程输出。
    返回:
        None。
    行为说明:
        1. stop 不谎报已退出；取消之后等待实际 join 完成。
    """
    batches = queue.Queue()
    batches.put((incoming("cancel"),))
    started, release = threading.Event(), threading.Event()
    runner = WeixinChannelRunner(
        tmp_path / "channels.sqlite3",
        lambda: FakeAdapter(batches),
        lambda **kwargs: FakeClient(**kwargs, started=started, release=release, cancelled=True),
        report=lambda _: None,
    )
    try:
        runner.start()
        assert started.wait(3)
        runner.stop()
        assert not runner.join(0.01)
        release.set()
        assert runner.join(3)
    finally:
        release.set()
        runner.stop()
        assert runner.join(5)
    assert runner.error is None
    with closing(ChannelStore(tmp_path / "channels.sqlite3")) as store:
        pending = store.pending_requests("weixin", "test-bot")
        assert len(pending) == 1 and pending[0].backend_request_id
        assert store.recover("weixin", "test-bot") == 0
    output = capsys.readouterr()
    assert "Traceback" not in output.err


def test_added_request_table_preserves_legacy_schema_and_cached_delivery(tmp_path):
    """验证纯新增关联表不改旧表、游标及投递标识。

    作用:
        现有微信数据不需要清空、重新导入或重放。
    参数:
        tmp_path: 隔离目录。
    返回:
        None。
    行为说明:
        1. 删除测试库中的空新表模拟旧版本，重开后比较旧表结构与记录。
    """
    path = tmp_path / "channels.sqlite3"
    item = incoming(str(2**63 - 1))
    with closing(ChannelStore(path)) as store:
        store.accept_batch("weixin", "test-bot", ReceiveBatch((item,), {"cursor": "legacy"}))
        store.bind(item, "legacy-session")
        record_id = store.next_queued().id
        store.claim(record_id, "legacy-session")
        store.prepare_reply(record_id, "缓存答案", ["缓存答案"], kind="model", turn_id="legacy-turn")
        before = store.get(record_id)
        original_schema = store.connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='channel_inbox'"
        ).fetchone()[0]
        store.connection.execute("DROP TABLE channel_request_links")
    with closing(ChannelStore(path)) as restored:
        assert restored.get(record_id) == before
        assert restored.checkpoint("weixin", "test-bot") == {"cursor": "legacy"}
        assert restored.binding(item) == "legacy-session"
        assert (
            restored.connection.execute("SELECT sql FROM sqlite_master WHERE name='channel_inbox'").fetchone()[0]
            == original_schema
        )
        assert restored.recover("weixin", "test-bot") == 0


def test_partial_thread_start_failure_stops_and_joins_started_worker(tmp_path, monkeypatch):
    """第二条线程启动失败时等待第一条线程真正退出。

    作用:
        防止外层误以为 runner 已清理，从而提前关闭应用服务或释放数据锁。
    参数:
        tmp_path: 隔离目录。
        monkeypatch: 仅令处理线程的 start 失败。
    返回:
        None。
    行为说明:
        1. join 忽略尚未启动的线程，失败回滚不留下后台线程。
    """
    original_start = threading.Thread.start

    def start_or_fail(thread):
        """启动接收线程并拒绝启动处理线程。

        作用:
            用真实线程覆盖部分启动成功的窗口。
        参数:
            thread: runner 正在启动的线程。
        返回:
            原 start 的结果。
        异常:
            RuntimeError: 当前为处理线程。
        行为说明:
            1. 不替换接收线程的实际执行和关闭逻辑。
        """
        if thread.name == "kairos-weixin-process":
            raise RuntimeError("合成线程启动失败")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start_or_fail)
    runner = WeixinChannelRunner(
        tmp_path / "channels.sqlite3", FakeAdapter, lambda **kwargs: FakeClient(**kwargs), report=lambda _: None
    )
    with pytest.raises(RuntimeError, match="合成线程启动失败"):
        runner.start()
    assert runner.stop_event.is_set()
    assert not runner.running and runner.join(0)
    assert runner._threads[0].ident is not None and runner._threads[1].ident is None
