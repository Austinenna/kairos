"""用离线模型和临时数据验证合作取消、回合关联与展示事件边界。"""

import copy
import json
from threading import Event, Thread, get_ident
from types import SimpleNamespace

import pytest

from kairos.cancellation import CancellationToken, RequestCancelled, cancellation_scope, raise_if_cancelled
from kairos.providers import AnthropicMessagesModel, OpenAICompatibleModel
from kairos.runtime import create_runtime
from kairos_todo import list_todos


class Model:
    """保留旧 complete 签名的模型替身。"""

    model = "offline-cancellation"

    def __init__(self, responses):
        """准备有限离线响应序列。

        作用:
            为取消和下一回合恢复提供可观测模型调用。
        参数:
            responses: 统一响应序列。
        返回:
            None。
        行为说明:
            1. 不初始化客户端或读取用户配置。
        """
        self.responses = iter(responses)
        self.requests = []

    def complete(self, *, messages, tools):
        """记录请求并返回一条完整响应。

        作用:
            验证取消实现没有要求旧注入模型增加参数。
        参数:
            messages: 模型上下文。
            tools: 工具定义。
        返回:
            下一条离线响应。
        行为说明:
            1. 深复制输入和输出，保持断言稳定。
        """
        self.requests.append(copy.deepcopy(messages))
        return copy.deepcopy(next(self.responses))


def call(call_id, name, **arguments):
    """构造供真实工具循环执行的调用。

    作用:
        让测试验证真实待办持久化，避免仅检查 mock 调用次数。
    参数:
        call_id: 本批唯一调用 ID。
        name: 工具名称。
        arguments: 工具参数。
    返回:
        统一工具调用字典。
    行为说明:
        1. 使用正常协议编码参数。
    """
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(arguments)}}


def test_token_is_thread_safe_and_context_does_not_leak():
    """验证取消可跨线程发起，但作用域只影响显式绑定的执行上下文。

    作用:
        防止一次任务取消污染下一任务或其他独立线程。
    返回:
        None。
    行为说明:
        1. 其他线程设置标记，本线程检查到取消。
        2. 新线程没有继承该任务作用域，退出作用域后本线程也恢复正常。
    """
    token = CancellationToken()
    thread = Thread(target=token.cancel)
    thread.start()
    thread.join(2)
    assert not thread.is_alive() and token.is_cancelled
    token.cancel()
    observed = []

    def independent():
        """在独立线程确认没有当前请求标记。

        作用:
            模拟不应继承前台取消状态的后台工作。
        返回:
            None。
        行为说明:
            1. 通过检查后记录线程已经正常完成。
        """
        raise_if_cancelled()
        observed.append(get_ident())

    with cancellation_scope(token):
        with pytest.raises(RequestCancelled):
            raise_if_cancelled()
        other = Thread(target=independent)
        other.start()
        other.join(2)
        assert not other.is_alive() and observed
    raise_if_cancelled()


def test_pre_cancelled_request_has_no_turn_and_does_not_poison_next_request(tmp_path):
    """验证执行前取消不创建回合，也不影响后续输入。

    作用:
        在 Runtime 最后一道准入处阻止已取消请求调用模型。
    参数:
        tmp_path: 隔离数据目录。
    返回:
        None。
    行为说明:
        1. 已取消标记被拒绝；旧 complete 模型在下一次请求正常工作。
    """
    model = Model([{"content": "下一轮正常"}])
    token = CancellationToken()
    token.cancel()
    with create_runtime(model=model, db_path=tmp_path / "sessions.sqlite3") as runtime:
        with pytest.raises(RequestCancelled):
            runtime.respond("不要执行", cancel_token=token)
        assert runtime.session_details()["turns"] == []
        assert model.requests == [] and not runtime.last_input_retained
        assert runtime.respond("下一轮", cancel_token=CancellationToken()).content == "下一轮正常"


def test_turn_start_callback_sees_durable_identity_before_any_model_call(tmp_path):
    """验证应用请求可以在模型执行前绑定到真实已保存的回合。

    作用:
        消除靠最近回合猜测请求归属的需要。
    参数:
        tmp_path: 隔离目录。
    返回:
        None。
    行为说明:
        1. 回调读取真实会话记录，确认 Turn 和用户消息已存在且模型尚未执行。
        2. 回调取得的标识与最终结果一致。
    """
    model = Model([{"content": "完成"}])
    started = []
    with create_runtime(model=model, db_path=tmp_path / "sessions.sqlite3") as runtime:

        def bind(info):
            """模拟应用层在执行前确认持久关联。

            作用:
                检查回调发生在写入 Turn 后、调用模型前。
            参数:
                info: session_id、turn_id、user_message_id。
            返回:
                None。
            行为说明:
                1. 检查实际数据库记录，不创建另一份虚构执行 ID。
            """
            assert model.requests == []
            details = runtime.session_details()
            assert details["turns"][0]["turn_id"] == info["turn_id"]
            assert details["turns"][0]["user_message_id"] == info["user_message_id"]
            assert details["turns"][0]["status"] == "running"
            assert details["steps"] == []
            started.append(dict(info))

        result = runtime.respond("建立关联", on_turn_start=bind)
        assert started == [
            {
                "session_id": result.session_id,
                "turn_id": result.turn_id,
                "user_message_id": runtime.session_details()["turns"][0]["user_message_id"],
            }
        ]


def test_failed_turn_start_callback_prevents_model_and_finishes_turn(tmp_path):
    """验证关联未确认时不允许执行模型，且不会留下活跃回合标记。

    作用:
        覆盖请求关联写入失败的清理路径。
    参数:
        tmp_path: 隔离目录。
    返回:
        None。
    行为说明:
        1. 回调失败后 Turn 为 failed、无模型步骤，后续会话操作仍可执行。
    """
    model = Model([{"content": "关联成功后的下一轮"}])

    def fail_bind(info):
        """模拟本机请求关联无法保存。

        作用:
            验证关联是实际执行的必要前置步骤。
        参数:
            info: 已持久保存的回合身份。
        返回:
            不正常返回。
        异常:
            RuntimeError: 无条件模拟关联保存失败。
        行为说明:
            1. 在任何模型或工具执行前抛出异常。
        """
        raise RuntimeError("synthetic binding failure")

    with create_runtime(model=model, db_path=tmp_path / "sessions.sqlite3") as runtime:
        with pytest.raises(RuntimeError, match="synthetic binding failure"):
            runtime.respond("关联失败的任务", on_turn_start=fail_bind)
        details = runtime.session_details()
        assert model.requests == [] and details["steps"] == []
        assert details["turns"][0]["status"] == "failed"
        assert not runtime.last_input_retained and not runtime._responding
        assert runtime.respond("下一轮").content == "关联成功后的下一轮"


def test_cancel_after_completed_tool_preserves_result_and_never_runs_remaining_tools(tmp_path):
    """验证取消发生在工具返回后时，保存真实结果并跳过剩余工具。

    作用:
        防止已完成副作用被误记为未知，或恢复时被再次执行。
    参数:
        tmp_path: 隔离目录。
    返回:
        None。
    行为说明:
        1. 第一条真实待办写入后请求取消，第二条不执行。
        2. 关闭并恢复会话后，已保存工具结果仍在，待办只存在一次。
    """
    database = tmp_path / "sessions.sqlite3"
    token = CancellationToken()
    model = Model(
        [{"tool_calls": [call("first", "add_todo", title="已执行"), call("second", "add_todo", title="不应执行")]}]
    )
    with create_runtime(model=model, db_path=database) as runtime:
        original = runtime.tool_registry.handlers["add_todo"]

        def save_then_cancel(**arguments):
            """执行真实工具后请求取消。

            作用:
                模拟用户在同步工具尚未返回到循环时点击取消。
            参数:
                arguments: 实际待办工具参数。
            返回:
                已提交的待办记录。
            行为说明:
                1. 副作用已经发生，仍正常返回结果供循环保存。
            """
            result = original(**arguments)
            token.cancel()
            return result

        runtime.tool_registry.handlers["add_todo"] = save_then_cancel
        with pytest.raises(RequestCancelled):
            runtime.respond("添加两条待办", cancel_token=token)
        session_id = runtime.session_id
        details = runtime.session_details()
        assert details["turns"][0]["status"] == "cancelled"
        assert [item["status"] for item in details["tool_calls"]] == ["completed", "not_executed"]
        assert [todo["title"] for todo in list_todos(runtime.connection)] == ["已执行"]
        assert len(model.requests) == 1 and runtime.last_input_retained

    restored_model = Model([{"content": "保留已经执行的记录"}])
    with create_runtime(model=restored_model, db_path=database, session_id=session_id) as restored:
        assert restored.respond("核对状态").content == "保留已经执行的记录"
        assert [todo["title"] for todo in list_todos(restored.connection)] == ["已执行"]
        results = [entry for entry in restored_model.requests[0] if entry["role"] == "tool"]
        assert json.loads(results[0]["content"])["title"] == "已执行"
        assert "未执行" in json.loads(results[1]["content"])["error"]["message"]


def test_cancel_control_does_not_pretend_to_interrupt_blocking_tool(tmp_path):
    """验证控制线程可请求取消，但正在运行的同步工具需先返回。

    作用:
        固定合作取消的实际边界，防止 UI 将已请求取消误报为已停止。
    参数:
        tmp_path: 隔离目录。
    返回:
        None。
    行为说明:
        1. 工作线程持有 Runtime，工具被测试事件暂停。
        2. 主线程取消后工具仍等待；释放工具后结果落盘，再停止回合。
    """
    token = CancellationToken()
    entered, release, finished = Event(), Event(), Event()
    outcome = []

    def execute():
        """在固定线程拥有完整 Runtime 生命周期。

        作用:
            真实模拟应用执行线程与取消控制线程分离。
        返回:
            None。
        行为说明:
            1. 仅捕获预期取消；无论成功与否都通知测试释放等待。
        """
        try:
            model = Model([{"tool_calls": [call("slow", "blocking")]}])
            with create_runtime(model=model, db_path=tmp_path / "sessions.sqlite3") as runtime:

                def blocking():
                    """等待测试释放后返回可保存的结果。

                    作用:
                        模拟不能被强制中断的同步工具。
                    返回:
                        工具完成结果。
                    行为说明:
                        1. 不检查取消标记，必须等当前操作结束。
                    """
                    entered.set()
                    assert release.wait(5)
                    return {"finished": True}

                runtime.tool_registry.handlers["blocking"] = blocking
                with pytest.raises(RequestCancelled):
                    runtime.respond("执行同步工具", cancel_token=token)
                outcome.append(runtime.session_details())
        finally:
            finished.set()

    worker = Thread(target=execute)
    worker.start()
    try:
        assert entered.wait(5)
        token.cancel()
        assert not finished.is_set()
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and len(outcome) == 1
    assert outcome[0]["turns"][0]["status"] == "cancelled"
    assert outcome[0]["tool_calls"][0]["status"] == "completed"


class CancelOnReadStream:
    """在指定无正文分片读取时设置取消标记的 SDK 流替身。"""

    def __init__(self, events, token, cancel_at):
        """准备流式取消场景。

        作用:
            验证取消检查依赖每个协议帧而不是正文回调。
        参数:
            events: SDK 事件序列。
            token: 当前请求取消标记。
            cancel_at: 从零开始的取消分片索引。
        返回:
            None。
        行为说明:
            1. 记录已读取事件数及关闭状态。
        """
        self.events = events
        self.token = token
        self.cancel_at = cancel_at
        self.read_count = 0
        self.closed = False

    def __iter__(self):
        """在一次模拟网络读取返回前请求取消。

        作用:
            将取消与 SDK 实际事件消费顺序联系起来。
        返回:
            逐项产生合成事件的迭代器。
        行为说明:
            1. 取消帧之后的事件不应被 provider 继续读取。
        """
        for index, event in enumerate(self.events):
            self.read_count += 1
            if index == self.cancel_at:
                self.token.cancel()
            yield event

    def close(self):
        """记录 provider 已释放流资源。

        作用:
            检查 KeyboardInterrupt 子类同样进入 finally。
        返回:
            None。
        行为说明:
            1. 不进行任何网络调用。
        """
        self.closed = True


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("phase", ["thinking", "tool_parameters", "empty"])
def test_no_text_stream_can_cancel_and_close_without_ui_callback(tmp_path, provider, phase):
    """验证两家 provider 在无正文阶段也能取消并关闭流。

    作用:
        避免微信等没有正文渲染回调的入口退回不可逐帧检查的 complete 请求。
    参数:
        tmp_path: 隔离目录。
        provider: 内置模型协议。
        phase: 思考、工具参数或空帧阶段。
    返回:
        None。
    行为说明:
        1. 在无正文帧设置取消，未完成的工具参数不进入执行或持久助手消息。
        2. 请求没有 on_event，仍使用内置流式接口观察取消。
    """
    token = CancellationToken()
    if provider == "openai":
        delta = {
            "thinking": {"reasoning_content": "未公开的思考"},
            "tool_parameters": {
                "tool_calls": [
                    {"index": 0, "id": "unfinished", "function": {"name": "add_todo", "arguments": '{"title":'}}
                ]
            },
            "empty": {},
        }[phase]
        events = [{"choices": [{"delta": delta}]}] * 3
        cancel_at = 1
    else:
        events = [{"type": "message_start", "message": {"content": []}}]
        events += {
            "thinking": [
                {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "思考"}},
            ],
            "tool_parameters": [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "tool_use", "id": "unfinished", "name": "add_todo", "input": {}},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"title":'},
                },
            ],
            "empty": [{"type": "ping"}, {"type": "ping"}],
        }[phase]
        cancel_at = len(events) - 1
        events.append({"type": "message_stop"})
    stream = CancelOnReadStream(events, token, cancel_at)
    requests = []

    def create(**request):
        """提供 SDK 的合成流响应。

        作用:
            检查 Runtime 确实选择了 stream=True。
        参数:
            request: provider 构造的 SDK 参数。
        返回:
            可关闭的流替身。
        行为说明:
            1. 保存参数，不访问外部服务。
        """
        requests.append(request)
        return stream

    if provider == "openai":
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        model = OpenAICompatibleModel(model="offline-cancellation", client=client)
    else:
        model = AnthropicMessagesModel(
            "offline-cancellation", client=SimpleNamespace(messages=SimpleNamespace(create=create))
        )
    with create_runtime(model=model, db_path=tmp_path / "sessions.sqlite3") as runtime:
        with pytest.raises(RequestCancelled):
            runtime.respond("在无正文阶段取消", cancel_token=token)
        assert requests[0]["stream"] is True
        assert stream.closed and stream.read_count == cancel_at + 1
        details = runtime.session_details()
        assert details["turns"][0]["status"] == "cancelled"
        assert details["tool_calls"] == [] and list_todos(runtime.connection) == []
        assert not runtime.last_input_retained


def test_display_events_exclude_private_context_and_payloads_but_logger_keeps_observation(tmp_path):
    """验证展示白名单与完整内部观察事件互不混用。

    作用:
        在多入口接入前阻止请求、思考和工具载荷进入公共事件。
    参数:
        tmp_path: 隔离目录。
    返回:
        None。
    行为说明:
        1. 请求、响应和工具都含合成私人内容，只有正文、阶段、关联与用量可展示。
        2. logger 仍接收完整观察，用量和工具结果汇总保持原语义。
    """
    model = Model(
        [
            {
                "tool_calls": [call("lookup", "lookup", query="PRIVATE_TOOL_ARGUMENT")],
                "reasoning_content": "PRIVATE_REASONING",
                "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
            },
            {"content": "可展示答案"},
        ]
    )
    visible, logged = [], []
    with create_runtime(model=model, db_path=tmp_path / "sessions.sqlite3") as runtime:
        runtime.logger = logged.append
        runtime.tool_registry.handlers["lookup"] = lambda query: {"found": "PRIVATE_TOOL_RESULT"}
        result = runtime.respond("PRIVATE_USER_CONTEXT", on_event=visible.append)
    public = json.dumps(visible, ensure_ascii=False)
    private = json.dumps(logged, ensure_ascii=False)
    for value in ("PRIVATE_USER_CONTEXT", "PRIVATE_REASONING", "PRIVATE_TOOL_ARGUMENT", "PRIVATE_TOOL_RESULT"):
        assert value not in public and value in private
    assert "可展示答案" in public
    assert any(event.get("usage", {}).get("total_tokens") == 11 for event in visible)
    assert all(not {"messages", "tools", "response", "arguments", "result"}.intersection(event) for event in visible)
    assert result.tool_results[0].value == {"found": "PRIVATE_TOOL_RESULT"}


def test_cancellation_from_model_response_event_stops_before_tools(tmp_path):
    """验证完整模型返回后、工具开始前仍会检查取消。

    作用:
        避免控制入口在工具启动边界取消却仍执行副作用。
    参数:
        tmp_path: 隔离目录。
    返回:
        None。
    行为说明:
        1. 接到模型响应阶段事件时取消，待办工具不执行。
    """
    token = CancellationToken()
    model = Model([{"tool_calls": [call("not-started", "add_todo", title="不要写入")]}])

    def observe(event):
        """在响应展示阶段模拟外部取消。

        作用:
            在模型完成与工具开始之间确定地设置取消事件。
        参数:
            event: 白名单展示事件。
        返回:
            None。
        行为说明:
            1. 不抛异常，只提交取消标记。
        """
        if event["event"] == "model_response":
            token.cancel()

    with create_runtime(model=model, db_path=tmp_path / "sessions.sqlite3") as runtime:
        with pytest.raises(RequestCancelled):
            runtime.respond("请求一个待办", cancel_token=token, on_event=observe)
        assert list_todos(runtime.connection) == []
        assert runtime.session_details()["turns"][0]["status"] == "cancelled"
