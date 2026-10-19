"""用真实统一程序、Runtime 和 SQLite 验证双入口与故障恢复；只替换外部输入输出。"""

import copy
import json
import queue
import sqlite3
import threading
from contextlib import closing
from types import SimpleNamespace

import pytest

from kairos import program
from kairos.app_lock import application_lock
from kairos.application import ApplicationService
from kairos.application_store import ApplicationStore
from kairos.channel_app import ChannelApp
from kairos.channels.base import InboundMessage, ReceiveBatch, SendReceipt
from kairos.channels.store import ChannelStore
from kairos.channels.weixin import write_credentials
from kairos.context import REFERENCE_HEADER
from kairos.runtime import create_runtime
from kairos_todo import list_todos


def message(message_id, text):
    """创建当前测试账号的微信消息。

    作用:
        将测试输入送入真实接收与去重流程。
    参数:
        message_id: 平台消息标识。
        text: 合成正文。
    返回:
        InboundMessage。
    行为说明:
        1. 账号、对话和发件人均是合成值。
    """
    return InboundMessage("weixin", "integration-bot", "integration-owner", "integration-owner", message_id, text)


class ScriptModel:
    """提供脚本响应并记录真实 Runtime 送入的上下文。"""

    model = "offline-integration-model"

    def __init__(self, responses, *, started=None, release=None):
        """准备离线模型响应及可选首轮阻塞。

        作用:
            模拟长模型任务，保留实际 Agent 循环、会话和工具行为。
        参数:
            responses: 按调用顺序返回的响应。
            started, release: 首次请求开始及释放事件。
        返回:
            无返回值。
        行为说明:
            1. 不创建 SDK 客户端或请求真实上游。
        """
        self.responses = iter(responses)
        self.requests = []
        self.threads = []
        self.started = started
        self.release = release

    def complete(self, *, messages, tools):
        """记录上下文并返回下一条脚本响应。

        作用:
            在真实工具和会话路径中观察模型执行次数及顺序。
        参数:
            messages: Runtime 构造的完整模型上下文。
            tools: 实际工具定义。
        返回:
            统一协议响应字典。
        行为说明:
            1. 首次请求可暂停，其他入口仍须能够持久提交和接收。
        """
        self.requests.append(copy.deepcopy(messages))
        self.threads.append(threading.get_ident())
        if len(self.requests) == 1 and self.started is not None:
            self.started.set()
            assert self.release.wait(5), "测试未在后台忙时完成另一入口提交"
        return copy.deepcopy(next(self.responses))


class Network:
    """模拟微信平台，连接器线程仍使用各自的适配器实例。"""

    def __init__(self):
        """准备合成入站队列及发送记录。

        作用:
            替代 iLink 网络，不替代 ChannelApp 或 ChannelStore。
        参数:
            无。
        返回:
            无返回值。
        行为说明:
            1. 每条出站记录保留原目标与文字，供检查结果路由。
        """
        self.batches = queue.Queue()
        self.sent = []
        self.second_sent = threading.Event()
        self.adapters = []

    def adapter(self, credentials=None):
        """创建当前线程私有的平台客户端。

        作用:
            与生产工厂保持一致，接收和发送不共享网络状态。
        参数:
            credentials: 入口读取的合成凭据，仅为兼容工厂调用。
        返回:
            新 Adapter。
        行为说明:
            1. 记录构造线程，不建立网络连接。
        """
        adapter = Adapter(self)
        self.adapters.append(adapter)
        return adapter


class Adapter:
    """当前线程持有的平台替身。"""

    channel, account_id, owner_id = "weixin", "integration-bot", "integration-owner"

    def __init__(self, network):
        """绑定外部平台模拟器。

        作用:
            区分网络实例所有权和共享平台消息。
        参数:
            network: 测试平台。
        返回:
            无返回值。
        行为说明:
            1. 保存当前线程用于收发时校验。
        """
        self.network = network
        self.thread = threading.get_ident()

    def poll(self, checkpoint):
        """有界等待下一批微信消息。

        作用:
            让真实接收线程持续运行并能响应停止信号。
        参数:
            checkpoint: 已保存平台游标。
        返回:
            新输入批次或保留游标的空批次。
        行为说明:
            1. 不读取聊天历史或私有账号。
        """
        assert threading.get_ident() == self.thread
        try:
            messages = self.network.batches.get(timeout=0.05)
        except queue.Empty:
            return ReceiveBatch((), checkpoint)
        return ReceiveBatch(tuple(messages), {"cursor": messages[-1].message_id})

    def send(self, target, text, client_id, checkpoint):
        """记录真实协调器准备的完整出站文字。

        作用:
            验证终端与微信各自得到正确会话的答案。
        参数:
            target, text, client_id, checkpoint: 实际通道投递参数。
        返回:
            明确业务成功的 SendReceipt。
        行为说明:
            1. 不向微信发送实际消息。
        """
        assert threading.get_ident() == self.thread
        self.network.sent.append((target, text, client_id))
        if text == "WECHAT-ANSWER-2":
            self.network.second_sent.set()
        return SendReceipt(True)


def runtime_factory(path, model, events, *, tools=()):
    """生成创建真实隔离 Runtime 的工厂。

    作用:
        观察业务对象是否只在应用工作线程创建、执行和关闭。
    参数:
        path: 测试数据根目录。
        model: 离线模型。
        events: 生命周期记录列表。
        tools: 明确启用的本地工具工厂。
    返回:
        可接受程序装配参数的 Runtime 工厂。
    行为说明:
        1. 所有业务数据库和记忆路径显式指定，自动记忆关闭以限制本测试范围。
    """

    def build(**kwargs):
        """创建并观察一个真实 Runtime。

        作用:
            使用真实执行引擎完成最终集成验证。
        参数:
            kwargs: program 注入的配置和路径，模型由测试明确提供。
        返回:
            AgentRuntime。
        行为说明:
            1. 仅包裹 close 记录线程，其余业务行为不替换。
        """
        events.append(("create", threading.get_ident()))
        runtime = create_runtime(
            model=model,
            db_path=path / "kairos.sqlite3",
            memory_dir=path / "memory",
            tool_factories=list(tools),
            auto_memory=False,
        )
        original_close = runtime.close

        def close():
            """记录真实 Runtime 的最终关闭线程。

            作用:
                检查入口没有自行关闭共享核心。
            参数:
                无。
            返回:
                None。
            行为说明:
                1. 委托原 close 完成资源收尾。
            """
            events.append(("close", threading.get_ident()))
            original_close()

        runtime.close = close
        return runtime

    return build


def test_program_two_entries_share_one_runtime_and_receive_while_busy(tmp_path, monkeypatch):
    """真实单进程程序实现微信 A、终端 B、微信 A 的连续交互。

    作用:
        同时验证唯一 Runtime、忙时持久接收与排队、会话历史及输出路由隔离。
    参数:
        tmp_path: 隔离用户数据目录。
        monkeypatch: 只替换模型、平台网络和终端输入驱动。
    返回:
        None。
    行为说明:
        1. 微信首轮停住后保存第二条入站，再由终端提交 B；真正提交完成才释放首轮。
        2. 程序结束后所有工作线程关闭，真实应用锁可以再次取得。
    """
    started, release, second_saved, terminal_queued = (threading.Event() for _ in range(4))
    model = ScriptModel(
        [{"content": "WECHAT-ANSWER-1"}, {"content": "TERMINAL-ANSWER"}, {"content": "WECHAT-ANSWER-2"}],
        started=started,
        release=release,
    )
    events, terminal_answers = [], []
    network = Network()
    network.batches.put((message("first", "WECHAT-A1"),))
    write_credentials(
        tmp_path / "channels" / "weixin" / "account.json",
        {
            "bot_token": "synthetic-token",
            "bot_id": "integration-bot",
            "owner_id": "integration-owner",
            "base_url": "https://ilinkai.weixin.qq.com",
            "initial_checkpoint": {},
        },
    )
    original_accept = ChannelStore.accept_batch
    original_submit = ApplicationStore.submit

    def accept_batch(store, channel, account, batch):
        """在第二条微信输入真正提交后设置检查事件。

        作用:
            不把网络收到替代为数据库已经保存。
        参数:
            store, channel, account, batch: 真实接收事务参数。
        返回:
            原实现返回的新增数量。
        行为说明:
            1. 原子保存行为完全由真实 ChannelStore 执行。
        """
        count = original_accept(store, channel, account, batch)
        if any(item.message_id == "second" for item in batch.messages):
            second_saved.set()
        return count

    def submit(store, request_id, source, operation, session_id, payload):
        """观察终端请求已持久接收但尚未执行的时刻。

        作用:
            证明模型忙时接收线程无需等模型完成才能接受另一入口输入。
        参数:
            store, request_id, source, operation, session_id, payload: 原应用请求参数。
        返回:
            原请求库的已保存记录及新增标记。
        行为说明:
            1. 只在真实提交成功后释放合成模型阻塞。
        """
        result = original_submit(store, request_id, source, operation, session_id, payload)
        if operation == "respond" and payload.get("prompt") == "TERMINAL-B":
            assert started.is_set() and second_saved.is_set() and not release.is_set()
            assert store.get(request_id)["status"] == "queued"
            terminal_queued.set()
            release.set()
        return result

    def terminal(*, runtime, notice_reader, allow_detach):
        """通过真实 RuntimeClient 提交合成终端输入。

        作用:
            仅替换交互输入和屏幕，不绕过应用服务执行。
        参数:
            runtime: 实际 terminal RuntimeClient。
            notice_reader: 主线程受控通知读取器。
            allow_detach: 微信已启用时的终端分离能力。
        返回:
            exit，触发真实 program 的统一收尾。
        行为说明:
            1. 等到微信两条答案投递后退出，避免提前取消后续任务。
        """
        assert allow_detach and started.wait(3)
        network.batches.put((message("second", "WECHAT-A2"),))
        assert second_saved.wait(3)
        with closing(ChannelStore(tmp_path / "channels" / "channels.sqlite3")) as store:
            assert store.checkpoint("weixin", "integration-bot") == {"cursor": "second"}
            assert store.next_queued().message.text == "WECHAT-A2"
        terminal_answers.append(runtime.respond("TERMINAL-B").content)
        assert network.second_sent.wait(3)
        return "exit"

    monkeypatch.setattr(program.config, "get_app_dir", lambda: tmp_path)
    monkeypatch.setattr(program, "create_runtime", runtime_factory(tmp_path, model, events))
    monkeypatch.setattr(program, "WeixinAdapter", network.adapter)
    monkeypatch.setattr(program.cli, "run", terminal)
    monkeypatch.setattr(ChannelStore, "accept_batch", accept_batch)
    monkeypatch.setattr(ApplicationStore, "submit", submit)
    try:
        assert program.run(channels=("terminal", "weixin"), settings=SimpleNamespace(kairos_api_key="synthetic")) == 0
    finally:
        release.set()
    assert terminal_queued.is_set()
    assert len(events) == 2 and [item[0] for item in events] == ["create", "close"]
    assert events[0][1] == events[1][1] != threading.get_ident()
    assert model.threads == [events[0][1]] * 3
    visible = [
        [
            item["content"]
            for item in request
            if item["role"] != "system" and not str(item.get("content", "")).startswith(REFERENCE_HEADER)
        ]
        for request in model.requests
    ]
    assert visible == [["WECHAT-A1"], ["TERMINAL-B"], ["WECHAT-A1", "WECHAT-ANSWER-1", "WECHAT-A2"]]
    assert terminal_answers == ["TERMINAL-ANSWER"]
    assert [item[1] for item in network.sent] == ["WECHAT-ANSWER-1", "WECHAT-ANSWER-2"]
    assert all(item[0].conversation_id == "integration-owner" for item in network.sent)
    assert len(network.adapters) == 2 and len({item.thread for item in network.adapters}) == 2
    assert not any(thread.is_alive() and thread.name.startswith("kairos-") for thread in threading.enumerate())
    with application_lock(tmp_path, label="离线收尾检查"):
        pass


def test_completed_turn_with_failed_request_commit_recovers_answer_without_repeating_tool(tmp_path, monkeypatch):
    """模型回合已落盘但应用结果保存失败时，重启只恢复原答案。

    作用:
        防止通道把未知状态改成假失败终态，从而丢失可恢复回答或重做副作用。
    参数:
        tmp_path: 隔离业务、请求与通道数据库目录。
        monkeypatch: 注入应用结果提交故障，不替换 Runtime 或工具实现。
    返回:
        None。
    行为说明:
        1. add_todo 真正写入一次；重启恢复完成 Turn，新的模型脚本不允许任何调用。
    """
    model = ScriptModel(
        [
            {
                "tool_calls": [
                    {"id": "once", "function": {"name": "add_todo", "arguments": json.dumps({"title": "只执行一次"})}}
                ]
            },
            {"content": "工具已完成，应该恢复这份原答案"},
        ]
    )
    events = []
    service = ApplicationService(
        runtime_factory(tmp_path, model, events, tools=("kairos_todo.tools:build_tools",)),
        request_path=tmp_path / "application" / "requests.sqlite3",
    )
    network = Network()
    store = ChannelStore(tmp_path / "channels.sqlite3")
    item = message("ledger-window", "创建一条待办")
    store.accept_batch("weixin", "integration-bot", ReceiveBatch((item,), {"cursor": "saved"}))
    app = ChannelApp(store, network.adapter(), service.client, report=lambda _: None)
    original_finish = service.store.finish

    def fail_turn_result(request_id, status, **kwargs):
        """在实际模型完成后拒绝保存应用结果。

        作用:
            重现业务 Turn 已完成而请求账本仍是 running 的故障窗口。
        参数:
            request_id, status, kwargs: 应用终态写入参数。
        返回:
            非目标操作委托原实现。
        异常:
            sqlite3.OperationalError: 保存已完成模型结果时。
        行为说明:
            1. 不影响此前的接收、Turn 关联或业务工具提交。
        """
        if status == "completed" and kwargs.get("result", {}).get("type") == "turn":
            raise sqlite3.OperationalError("合成应用结果提交故障")
        return original_finish(request_id, status, **kwargs)

    monkeypatch.setattr(service.store, "finish", fail_turn_result)
    try:
        record_id = store.next_queued().id
        with pytest.raises(sqlite3.OperationalError):
            app.process_next()
        record = store.get(record_id)
        assert record.status == "processing" and record.response is None and record.backend_request_id
        request_id = record.backend_request_id
        assert service.store.get(request_id)["status"] == "running"
        assert service.store.get(request_id)["turn_id"] is not None
    finally:
        app.close()
        store.close()
        service.close()
    assert len(model.requests) == 2
    with closing(sqlite3.connect(tmp_path / "kairos.sqlite3")) as connection:
        connection.row_factory = sqlite3.Row
        assert len(list_todos(connection)) == 1

    never_called = ScriptModel([])
    restored_service = ApplicationService(
        runtime_factory(tmp_path, never_called, [], tools=("kairos_todo.tools:build_tools",)),
        request_path=tmp_path / "application" / "requests.sqlite3",
    )
    with closing(ChannelStore(tmp_path / "channels.sqlite3")) as restored_store:
        restored_app = ChannelApp(restored_store, network.adapter(), restored_service.client, report=lambda _: None)
        try:
            assert restored_app.process_next()
            restored = restored_store.get(record_id)
            assert restored.response == "工具已完成，应该恢复这份原答案"
            assert restored.kind == "model" and restored.backend_request_id == request_id
            assert restored_app.flush_replies() == 1
            assert network.sent[-1][1] == restored.response
        finally:
            restored_app.close()
            restored_service.close()
    assert never_called.requests == []
    with closing(sqlite3.connect(tmp_path / "kairos.sqlite3")) as connection:
        connection.row_factory = sqlite3.Row
        assert len(list_todos(connection)) == 1
