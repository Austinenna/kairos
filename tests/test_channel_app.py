"""用真实临时 Runtime、会话、待办与记忆验证通道，只替换网络和模型。"""

import copy
import json
import queue
import re
from collections import deque
from contextlib import closing
from dataclasses import replace
from threading import get_ident
from types import SimpleNamespace

import pytest

from kairos import program, weixin
from kairos.channel_app import ChannelApp, ReplyDeliveryError, split_reply
from kairos.channels.base import InboundMessage, ReceiveBatch, SendReceipt, TransportError
from kairos.channels.store import INTERRUPTED_REPLY, ChannelStore
from kairos.context import REFERENCE_HEADER
from kairos.legacy_memory import MemoryStore
from kairos.runtime import create_runtime
from kairos_todo import list_todos


class FakeModel:
    """返回预定模型响应并记录实际上下文与执行线程。"""

    model = "channel-offline-model"

    def __init__(self, responses):
        """准备一次离线模型脚本。

        作用:
            为真实 Agent 循环提供确定的文字或工具调用。
        参数:
            responses: 依序返回的统一模型响应。
        返回:
            None。
        行为说明:
            1. 不初始化客户端，不读取用户配置。
        """
        self.responses = iter(responses)
        self.requests = []
        self.threads = []

    def complete(self, *, messages, tools):
        """记录请求并返回下一条离线响应。

        作用:
            让测试核对历史隔离及模型实际调用次数。
        参数:
            messages: Runtime 组装的完整上下文。
            tools: Runtime 注册的真实工具定义。
        返回:
            一条统一模型响应。
        行为说明:
            1. 深复制请求和响应，避免后续上下文变化污染断言。
        """
        self.threads.append(get_ident())
        self.requests.append(copy.deepcopy({"messages": messages, "tools": tools}))
        return copy.deepcopy(next(self.responses))


class FakeAdapter:
    """提供获准消息批次，并模拟某次发送结果未知。"""

    channel = "weixin"
    account_id = "synthetic-bot"
    owner_id = "synthetic-owner"

    def __init__(self, batches=(), *, fail_sends=()):
        """保存网络替身的收发脚本。

        作用:
            避免集成测试访问 iLink 或读取登录状态。
        参数:
            batches: 每次轮询返回的消息序列。
            fail_sends: 需要模拟失败的发送序号，从 1 开始。
        返回:
            None。
        行为说明:
            1. 分别记录尝试发送与接口确认成功的发送。
        """
        self.batches = deque(batches)
        self.fail_sends = set(fail_sends)
        self.checkpoints = []
        self.attempts = []
        self.sent = []

    def poll(self, checkpoint):
        """返回下一批合成输入与检查点。

        作用:
            检查协调器通过持久检查点恢复拉取。
        参数:
            checkpoint: 上次保存的适配器状态。
        返回:
            ReceiveBatch。
        行为说明:
            1. 脚本耗尽会使测试立即失败，避免错误循环永久等待。
        """
        self.checkpoints.append(copy.deepcopy(checkpoint))
        return ReceiveBatch(tuple(self.batches.popleft()), {"cursor": f"cursor-{len(self.checkpoints)}"})

    def send(self, target, text, client_id, checkpoint):
        """记录发送，并按脚本返回成功或结果未知。

        作用:
            将网络投递失败与模型、工具执行失败区分开。
        参数:
            target: 统一回复目标。
            text: 稳定分段正文。
            client_id: 本段已保存的请求标识。
            checkpoint: 当前持久适配器状态。
        返回:
            接口明确接受的 SendReceipt。
        异常:
            TransportError: 当前发送序号属于预设失败集合。
        行为说明:
            1. 失败请求也计入 attempts，模拟请求可能已到平台但没有可靠响应。
        """
        attempt = {"target": target, "text": text, "client_id": client_id, "checkpoint": copy.deepcopy(checkpoint)}
        self.attempts.append(attempt)
        if len(self.attempts) in self.fail_sends:
            raise TransportError("合成发送故障", uncertain=True)
        self.sent.append(attempt)
        return SendReceipt(acknowledged=True)


class RuntimeFactory:
    """创建真实隔离 Runtime，并观察前台生命周期的线程归属。"""

    def __init__(self, data_dir, model, *, extraction_factory=None):
        """设置测试数据位置与模型替身。

        作用:
            让所有 Runtime 使用同一测试数据库和记忆，以覆盖恢复行为。
        参数:
            data_dir: pytest 临时目录。
            model: 离线前台模型。
            extraction_factory: 可选离线后台模型工厂。
        返回:
            None。
        行为说明:
            1. 只保存依赖，尚不创建数据库或线程。
        """
        self.data_dir = data_dir
        self.model = model
        self.extraction_factory = extraction_factory
        self.runtimes = []
        self.events = []
        self.session_ids = []
        self.options = []

    def __call__(self, *, session_id=None, **options):
        """构造真实 Runtime 并观察其 respond 与 close。

        作用:
            测试保留真实工具、持久历史和后台提取，只记录前台生命周期。
        参数:
            session_id: 通道绑定需要恢复的会话。
            options: 微信 main 传入的路径与设置，仅记录供断言。
        返回:
            真实 AgentRuntime。
        行为说明:
            1. 采用明确的临时路径和工具工厂，禁止访问真实用户数据。
            2. 包装方法仍委托原实现，关闭和写库行为不被替换。
        """
        self.events.append(("create", get_ident()))
        self.session_ids.append(session_id)
        self.options.append(options)
        runtime = create_runtime(
            model=self.model,
            db_path=self.data_dir / "kairos.sqlite3",
            memory_dir=self.data_dir / "memory",
            session_id=session_id,
            tool_factories=["kairos_todo.tools:build_tools"],
            auto_memory=self.extraction_factory is not None,
            extraction_model_factory=self.extraction_factory,
        )
        original_respond, original_close, original_new_session = runtime.respond, runtime.close, runtime.new_session

        def respond(text, *, request_id=None, **kwargs):
            """在同一线程记录并执行真实回合。

            作用:
                核对协调器没有把已有 SQLite 连接交给其他线程。
            参数:
                text: 用户输入。
                request_id: 通道生成的稳定 ID；这个离线 Runtime 注入测试只验证原执行流程。
                kwargs: 应用服务传入的图像、事件、取消与回合关联参数。
            返回:
                原 Runtime 的回合结果。
            行为说明:
                1. 记录线程，原样调用真实 respond。
            """
            self.events.append(("respond", get_ident()))
            return original_respond(text, **kwargs)

        def new_session(*, title=None, system_prompt=None, request_id=None):
            """用客户端兼容签名委托真实离线会话创建。

            作用:
                让既有业务回归继续覆盖真实 Session 行为，应用幂等另由集成测试检查。
            参数:
                title: 可选新会话标题。
                system_prompt: 可选会话提示词。
                request_id: 通道稳定请求 ID，本离线包装不承担后台请求账本。
            返回:
                原 Runtime 创建的 SessionInfo。
            行为说明:
                1. 不替换实际会话存储。
            """
            return original_new_session(title=title, system_prompt=system_prompt)

        def close():
            """记录关闭线程并完成真实后台收尾。

            作用:
                核对 Runtime 完整生命周期的线程一致性。
            返回:
                None。
            行为说明:
                1. 记录线程，调用原 close 等待记忆并关闭数据库。
            """
            self.events.append(("close", get_ident()))
            original_close()

        runtime.respond = respond
        runtime.new_session = new_session
        runtime.close = close
        self.runtimes.append(runtime)
        return runtime


def message(message_id, text):
    """生成本人私聊的合成输入。

    作用:
        为不同场景提供稳定的外部身份。
    参数:
        message_id: 平台消息 ID。
        text: 消息正文，None 表示不支持的媒体。
    返回:
        InboundMessage。
    行为说明:
        1. 账号、会话和发件人均为合成值。
    """
    return InboundMessage("weixin", "synthetic-bot", "synthetic-owner", "synthetic-owner", message_id, text)


def tool(name, **arguments):
    """生成进入真实工具循环的响应。

    作用:
        通过实际工具分发产生待办或记忆副作用。
    参数:
        name: 工具名。
        arguments: 工具参数。
    返回:
        包含单个工具调用的响应字典。
    行为说明:
        1. 按统一协议序列化参数。
    """
    return {"tool_calls": [{"id": "synthetic-call", "function": {"name": name, "arguments": json.dumps(arguments)}}]}


@pytest.fixture
def build_app(tmp_path):
    """组装可重启的真实存储和通道应用。

    作用:
        每个测试复用自己的持久文件，测试间完全隔离。
    参数:
        tmp_path: 隔离数据目录。
    返回:
        创建应用、网络替身、存储和 Runtime 工厂的函数。
    行为说明:
        1. 无论断言成功或失败，都在测试末尾释放应用和存储。
    """
    opened = []

    def build(model, batches=(), *, fail_sends=(), reply_byte_limit=1800):
        """创建一轮应用实例。

        作用:
            通过再次调用模拟应用重启。
        参数:
            model: 离线模型。
            batches: 合成接收批次。
            fail_sends: 预设失败的发送序号。
            reply_byte_limit: 每段回复字节预算。
        返回:
            应用、适配器、存储和工厂四元组。
        行为说明:
            1. 所有状态均位于当前测试目录。
        """
        adapter = FakeAdapter(batches, fail_sends=fail_sends)
        store = ChannelStore(tmp_path / "channels.sqlite3")
        factory = RuntimeFactory(tmp_path, model)
        app = ChannelApp(store, adapter, factory, report=lambda _: None, reply_byte_limit=reply_byte_limit)
        opened.append((app, store))
        return app, adapter, store, factory

    yield build
    for app, store in reversed(opened):
        app.close()
        store.close()


def test_multiturn_session_and_duplicate_ids_survive_restart(build_app):
    """验证连续对话与进程重启都精确恢复同一个绑定。

    作用:
        同时检查上下文连续性和平台重复投递不重复生成答案。
    参数:
        build_app: 隔离应用工厂。
    返回:
        None。
    行为说明:
        1. 首次运行处理两条独立输入，重复 ID 不增加回合。
        2. 关闭后重建应用，恢复原会话并只处理新输入。
    """
    first, second, third = message("1", "第一问"), message("2", "第二问"), message("3", "第三问")
    model = FakeModel([{"content": "第一答"}, {"content": "第二答"}])
    app, adapter, store, factory = build_app(model, [(first, first, second)])
    assert app.run(max_messages=2) == 2
    session_id = store.binding(first)
    assert session_id and len(factory.runtimes) == 1
    assert len(model.requests) == len(adapter.sent) == 2
    store.close()

    restored_model = FakeModel([{"content": "第三答"}])
    restored, network, resumed_store, resumed_factory = build_app(restored_model, [(second, third)])
    assert restored.run(max_messages=1) == 1
    assert resumed_store.binding(third) == session_id
    assert resumed_factory.session_ids == [session_id]
    assert len(network.sent) == len(restored_model.requests) == 1
    assert network.checkpoints == [{"cursor": "cursor-1"}]
    conversation = [
        item
        for item in restored_model.requests[0]["messages"]
        if item["role"] != "system" and not str(item.get("content", "")).startswith(REFERENCE_HEADER)
    ]
    assert [item["content"] for item in conversation] == ["第一问", "第一答", "第二问", "第二答", "第三问"]


def test_new_session_separates_history_and_keeps_real_memory(build_app):
    """验证 /new 切换绑定、保留历史并共享真实记忆。

    作用:
        区分平台对话、内部会话和跨会话记忆的三种生命周期。
    参数:
        build_app: 隔离应用工厂。
    返回:
        None。
    行为说明:
        1. 通过 remember 工具写入偏好，再在同一应用执行 /new。
        2. 新请求只含新会话历史和记忆参考，旧会话完整保留。
    """
    fact = "交流时先说明整体。"
    first = message("1", "请记住我的偏好，旧话题标记 ALPHA")
    switch, followup = message("2", "/new 新话题"), message("3", "开始新问题 BETA")
    model = FakeModel([tool("remember", content=fact), {"content": "偏好已保存"}, {"content": "新问题回答"}])
    app, adapter, store, factory = build_app(model, [(first, switch, followup)])
    assert app.poll_once() == 3
    assert app.process_next()
    old_id = store.binding(first)
    assert app.flush_replies() == 1
    assert app.process_next()
    new_id = store.binding(switch)
    assert old_id != new_id
    assert app.process_next()
    app.flush_replies()
    runtime = factory.runtimes[0]
    assert runtime.session_id == new_id and runtime.session_info.title == "新话题"
    assert len(factory.runtimes) == 1
    latest = json.dumps(model.requests[-1]["messages"], ensure_ascii=False)
    assert fact in latest and "BETA" in latest and "ALPHA" not in latest
    old = runtime.session_store.inspect(old_id)
    assert any("ALPHA" in str(item["payload"]) for item in old["messages"])
    assert fact in runtime.memory_store.snapshot()[0]
    assert len(adapter.sent) == 3


def test_local_commands_and_unsupported_input_never_call_model(build_app):
    """验证通道管理与不支持的输入只产生本地回复。

    作用:
        防止帮助、会话命令和媒体意外触发模型或业务工具。
    参数:
        build_app: 隔离应用工厂。
    返回:
        None。
    行为说明:
        1. 连续处理帮助、会话、未知命令、错误参数、媒体、空白和无缓存重发。
    """
    texts = ["/session", "/help", "/unknown", "/help invalid", None, "  ", "/retry"]
    model = FakeModel([])
    app, adapter, store, factory = build_app(model, [tuple(message(str(i), text) for i, text in enumerate(texts))])
    assert app.run(max_messages=len(texts)) == len(texts)
    assert model.requests == []
    assert len(adapter.sent) == len(texts)
    replies = [item["text"] for item in adapter.sent]
    assert "当前会话：" in replies[0]
    assert "/new" in replies[1]
    assert "命令或参数不支持" in replies[2] and "命令或参数不支持" in replies[3]
    assert "只支持纯文字" in replies[4]
    assert "非空" in replies[5]
    assert "没有已生成" in replies[6]
    assert len(factory.runtimes) == 1
    assert store.next_queued() is None


def test_todo_executes_once_even_when_delivery_fails_restart_and_retry(build_app):
    """验证真实待办副作用不会因投递失败、重复消息或显式重发而重做。

    作用:
        覆盖通道最关键的执行与发送分离边界。
    参数:
        build_app: 隔离应用工厂。
    返回:
        None。
    行为说明:
        1. 工具真正写入待办，发送失败后关闭全部连接。
        2. 重启不自动发送未知结果；重复 ID 被忽略，/retry 只发缓存答案。
    """
    original = message("task", "添加一条仅执行一次的待办")
    answer = "已创建待办：仅执行一次。"
    model = FakeModel([tool("add_todo", title="仅执行一次"), {"content": answer}])
    app, adapter, store, factory = build_app(model, [(original,)], fail_sends=(1,))
    assert app.poll_once() == 1
    record_id = store.next_queued().id
    assert app.process_next()
    runtime = factory.runtimes[0]
    todos = list_todos(runtime.connection)
    assert len(todos) == 1 and todos[0]["title"] == "仅执行一次"
    with pytest.raises(ReplyDeliveryError):
        app.flush_replies()
    assert store.get(record_id).status == "delivery_uncertain"
    assert len(adapter.attempts) == 1 and len(model.requests) == 2
    app.close()
    store.close()

    restart_model = FakeModel([])
    retry = message("retry", "/retry")
    restored, network, resumed_store, resumed_factory = build_app(restart_model, [(original, retry)])
    assert restored.flush_replies() == 0
    assert resumed_factory.runtimes == [] and network.attempts == []
    assert restored.poll_once() == 1
    assert restored.process_next()
    assert restored.flush_replies() == 1
    assert network.sent[0]["text"] == answer
    assert restart_model.requests == []
    assert list_todos(resumed_factory.runtimes[0].connection) == todos
    assert resumed_store.get(record_id).status == "delivery_uncertain"
    assert network.sent[0]["client_id"] != adapter.attempts[0]["client_id"]


def test_run_keeps_receiving_explicit_retry_after_uncertain_send(build_app):
    """验证一次未知投递不会阻止用户发送重发命令。

    作用:
        检查长期运行循环可以继续接收，但不自动重新执行或发送。
    参数:
        build_app: 隔离应用工厂。
    返回:
        None。
    行为说明:
        1. 首次发送失败，下一批 /retry 成功，模型只调用一次。
    """
    model = FakeModel([{"content": "唯一生成的回答"}])
    app, adapter, _, _ = build_app(
        model, [(message("question", "问题"),), (message("retry", "/retry"),)], fail_sends=(1,)
    )
    assert app.run(max_messages=2) == 2
    assert len(model.requests) == 1
    assert len(adapter.checkpoints) == 2 and len(adapter.attempts) == 2
    assert [item["text"] for item in adapter.sent] == ["唯一生成的回答"]


def test_interrupted_processing_only_sends_notice_without_runtime(tmp_path, build_app):
    """验证遗留处理中输入只转为中断通知。

    作用:
        结果未知时不自动恢复执行可能已有副作用的任务。
    参数:
        tmp_path: 隔离目录。
        build_app: 隔离应用工厂。
    返回:
        None。
    行为说明:
        1. 写入 processing 后关闭存储模拟崩溃，再次启动只发通知且不建 Runtime。
    """
    with closing(ChannelStore(tmp_path / "channels.sqlite3")) as initial:
        incoming = message("interrupted", "可能已经执行部分操作")
        initial.accept_batch("weixin", "synthetic-bot", ReceiveBatch((incoming,), {"cursor": "saved"}))
        record_id = initial.next_queued().id
        initial.bind(incoming, "synthetic-interrupted-session")
        initial.claim(record_id, "synthetic-interrupted-session")
    app, adapter, store, factory = build_app(FakeModel([]))
    assert app.flush_replies() == 1
    assert adapter.sent[0]["text"] == INTERRUPTED_REPLY
    assert factory.runtimes == [] and not app.process_next()
    record = store.get(record_id)
    assert record.status == "sent" and record.kind == "error" and record.error_code == "interrupted"


@pytest.mark.parametrize("byte_limit", [64, 1800])
def test_split_reply_preserves_all_unicode_and_respects_budget(byte_limit):
    """验证长回复在最小与常用预算下完整保留 Unicode 正文。

    作用:
        防止中文、组合字符、表情或换行在分段中丢失。
    参数:
        byte_limit: 包含编号的单段字节预算。
    返回:
        None。
    行为说明:
        1. 去除顺序编号后重组全文，同时检查每段 UTF-8 字节数。
    """
    original = "开头\n" + "🙂e\u0301家庭👨‍👩‍👧‍👦结束\n" * 200
    chunks = split_reply(original, byte_limit)
    assert len(chunks) > 1
    assert all(len(chunk.encode("utf-8")) <= byte_limit for chunk in chunks)
    assert "".join(re.sub(r"^（\d+/\d+）\n", "", chunk) for chunk in chunks) == original
    assert all(chunk.startswith(f"（{index}/{len(chunks)}）\n") for index, chunk in enumerate(chunks, 1))


def test_partial_send_unknown_stops_remaining_parts_and_never_repeats(build_app):
    """验证长回答部分发送后发生未知结果时不会擅自补发。

    作用:
        已送达、未知和未开始的分段需要共同保持原投递状态。
    参数:
        build_app: 隔离应用工厂。
    返回:
        None。
    行为说明:
        1. 第一段成功、第二段未知，后续片段暂停；重新启动仍不自动发送。
    """
    answer = "需要完整保留的长回答🙂。\n" * 80
    app, adapter, store, _ = build_app(
        FakeModel([{"content": answer}]), [(message("long", "请详细回答"),)], fail_sends=(2,), reply_byte_limit=96
    )
    app.poll_once()
    record_id = store.next_queued().id
    app.process_next()
    with pytest.raises(ReplyDeliveryError):
        app.flush_replies()
    record = store.get(record_id)
    assert len(record.parts) > 2 and record.response == answer
    assert [part["status"] for part in record.parts[:2]] == ["submitted", "uncertain"]
    assert all(part["status"] == "ready" for part in record.parts[2:])
    assert app.flush_replies() == 0 and len(adapter.attempts) == 2
    app.close()
    store.close()

    restarted, network, recovered, factory = build_app(FakeModel([]))
    assert restarted.flush_replies() == 0
    assert network.attempts == [] and factory.runtimes == []
    assert recovered.get(record_id).parts == record.parts


def test_weixin_entry_uses_application_worker_and_finishes_memory(tmp_path, monkeypatch):
    """验证真实微信入口经统一应用线程执行，并在退出前完成后台记忆写入。

    作用:
        覆盖 main 的依赖组装、运行锁以及 Runtime 资源生命周期。
    参数:
        tmp_path: 隔离目录。
        monkeypatch: 仅替换微信网络、配置加载及模型工厂。
    返回:
        None。
    行为说明:
        1. main 处理一条消息后退出；Runtime 创建、执行、关闭均在专用应用线程。
        2. 真后台提取通过 write_core 写入临时记忆，退出时线程已结束。
    """
    fact = "以后交流默认用中文。"
    foreground = FakeModel([{"content": "好的"}])
    extraction = FakeModel([tool("write_core", content=fact), {"content": '{"status":"completed"}'}])
    factory = RuntimeFactory(tmp_path, foreground, extraction_factory=lambda: extraction)
    batches = queue.Queue()
    batches.put((message("memory", fact),))
    delivered = []

    class ThreadAdapter(FakeAdapter):
        """为接收和发送线程各自创建平台替身。"""

        def __init__(self):
            """准备独立实例与共享的合成发送结果。

            作用:
                不共享真实 HTTP 客户端，同时允许测试检查最终手机输出。
            参数:
                无。
            返回:
                无返回值。
            行为说明:
                1. 只共享测试列表和消息队列。
            """
            super().__init__()
            self.sent = delivered

        def poll(self, checkpoint):
            """等待一条合成消息，队列耗尽后返回空批次。

            作用:
                模拟可以安全停止的持续长轮询。
            参数:
                checkpoint: 已持久化游标。
            返回:
                ReceiveBatch。
            行为说明:
                1. 不连接微信，不在模型执行期间抛出脚本耗尽错误。
            """
            try:
                items = batches.get(timeout=0.05)
            except queue.Empty:
                return ReceiveBatch((), checkpoint)
            return ReceiveBatch(items, {"cursor": "saved"})

    account = tmp_path / "channels" / "weixin" / "account.json"
    account.parent.mkdir(parents=True)
    account.write_text("{}")
    monkeypatch.setattr(program.config, "get_app_dir", lambda: tmp_path)
    monkeypatch.setattr(program, "Settings", lambda: SimpleNamespace(kairos_api_key="synthetic-key"))
    monkeypatch.setattr(program, "read_credentials", lambda _: {})
    monkeypatch.setattr(program, "WeixinAdapter", lambda _: ThreadAdapter())
    monkeypatch.setattr(program, "create_runtime", factory)

    assert weixin.main(["run", "--max-messages", "1"]) == 0
    worker_thread = factory.events[0][1]
    assert worker_thread != get_ident()
    assert factory.events == [(name, worker_thread) for name in ("create", "respond", "close")]
    assert foreground.threads == [worker_thread]
    assert extraction.threads and all(thread not in (get_ident(), worker_thread) for thread in extraction.threads)
    assert factory.options[0]["db_path"] == tmp_path / "kairos.sqlite3"
    assert factory.options[0]["memory_dir"] == tmp_path / "memory"
    runtime = factory.runtimes[0]
    assert runtime._closed and not runtime.memory_extractor._worker.is_alive()
    assert fact in MemoryStore(tmp_path / "memory").snapshot()[0]
    assert len(delivered) == 1


def test_only_owner_current_account_messages_enter_runtime(build_app):
    """验证授权过滤与积压账号隔离共同阻止其他消息进入模型。

    作用:
        保护单用户工具与全局记忆不被外来输入调用。
    参数:
        build_app: 隔离应用工厂。
    返回:
        None。
    行为说明:
        1. 批次混入非本人、其他平台、其他账号消息，只有合法输入被处理。
        2. 旧账号已有的 queued 输入保持未处理，不阻挡当前账号。
    """
    accepted = message("allowed", "获准的本人输入")
    rejected = [
        replace(accepted, message_id="stranger", sender_id="other-person"),
        replace(accepted, message_id="other-platform", channel="other-platform"),
        replace(accepted, message_id="other-account", account_id="other-account"),
    ]
    model = FakeModel([{"content": "只回复本人"}])
    app, adapter, store, _ = build_app(model, [(*rejected, accepted)])
    old = replace(accepted, account_id="old-account", message_id="old-queued")
    store.accept_batch("weixin", "old-account", ReceiveBatch((old,), {}))
    assert app.run(max_messages=1) == 1
    assert len(model.requests) == len(adapter.sent) == 1
    assert adapter.sent[0]["target"].conversation_id == accepted.conversation_id
    assert store.next_queued("weixin", "old-account").message == old
    assert store.next_queued("weixin", "synthetic-bot") is None
    saved = store.connection.execute("SELECT account_id, sender_id FROM channel_inbox ORDER BY id").fetchall()
    assert [tuple(row) for row in saved] == [("old-account", "synthetic-owner"), ("synthetic-bot", "synthetic-owner")]
