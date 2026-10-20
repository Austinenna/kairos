"""离线验证每用户回合的本机时间参考及两种模型协议边界。"""

import copy
import json
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import kairos.runtime as runtime_module
from kairos.providers import AnthropicMessagesModel, OpenAICompatibleModel
from kairos.runtime import create_runtime


@pytest.fixture(params=[("Asia/Shanghai", "+08:00", "CST"), ("America/New_York", "-04:00", "EDT")])
def local_clock(request, monkeypatch):
    """固定本机日期并切换测试进程的本地时区。

    作用:
        让运行时仍通过标准库 astimezone 解析本机时区，验证实现没有固定使用东八区。

    参数:
        request: pytest 提供的时区名、ISO 偏移和时区缩写参数。
        monkeypatch: pytest 的临时替换工具。

    返回:
        可推进的时钟、取时 Mock、预期偏移和时区缩写 (SimpleNamespace)。

    行为说明:
        1. 只替换 runtime 引用的 datetime.now，保留真实 datetime.astimezone。
        2. 测试完成或失败后恢复 TZ 环境变量，并重新载入原本的本地时区。
    """
    zone, offset, name = request.param
    clock = SimpleNamespace(value=datetime(2026, 10, 20, 23, 59, 30, 123456), offset=offset, name=name)
    clock.now = Mock(side_effect=lambda: clock.value)
    try:
        with monkeypatch.context() as patch:
            patch.setenv("TZ", zone)
            time.tzset()
            patch.setattr(runtime_module, "datetime", SimpleNamespace(now=clock.now))
            yield clock
    finally:
        time.tzset()


def offline_model(provider, actions):
    """为真实协议适配器安装返回固定响应的离线客户端。

    作用:
        捕获实际发送给客户端的请求，覆盖 system 在两种协议中的不同承载位置。

    参数:
        provider: openai 或 anthropic。
        actions: 依次返回的工具调用 ID；None 表示最终文字回答。

    返回:
        真实模型适配器和实际请求记录 (tuple)。

    行为说明:
        1. 按适配器请求的 stream 参数生成完整响应或可关闭的离线事件流。
        2. 工具只推进测试时钟，不访问网络或真实个人数据。
    """
    replies = iter(actions)
    requests = []

    def create(**request):
        """记录协议请求并返回本次预设的模型响应。

        作用:
            替代 SDK 的 create 调用，让完整运行时执行真实适配器的请求转换。

        参数:
            request: 真实适配器组装的模型请求参数。

        返回:
            协议响应字典或可关闭的流式响应生成器。

        行为说明:
            1. 深拷贝请求，避免后续回合改写已捕获的证据。
            2. 将预设动作转换成对应协议的工具调用或最终文字。
        """
        requests.append(copy.deepcopy(request))
        call_id = next(replies)
        if provider == "openai":
            message = (
                {
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "advance_clock", "arguments": "{}"},
                        }
                    ]
                }
                if call_id
                else {"content": "已完成"}
            )
            reason = "tool_calls" if call_id else "stop"
            if not request.get("stream"):
                return {"choices": [{"message": message, "finish_reason": reason}]}
            if call_id:
                message["tool_calls"][0]["index"] = 0
            frames = [
                {"choices": [{"index": 0, "delta": message, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]},
            ]
        else:
            block = (
                {"type": "tool_use", "id": call_id, "name": "advance_clock", "input": {}}
                if call_id
                else {"type": "text", "text": "已完成"}
            )
            reason = "tool_use" if call_id else "end_turn"
            if not request.get("stream"):
                return {"content": [block], "stop_reason": reason}
            frames = [
                {"type": "message_start", "message": {"content": []}},
                {"type": "content_block_start", "index": 0, "content_block": block},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": reason}},
                {"type": "message_stop"},
            ]
        return (frame for frame in frames)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        messages=SimpleNamespace(create=create),
    )
    model = (
        OpenAICompatibleModel(model="time-test-model", client=client)
        if provider == "openai"
        else AnthropicMessagesModel("time-test-model", client=client)
    )
    return model, requests


def system_text(request, provider):
    """读取实际协议请求中的 system 文本。

    作用:
        统一两种协议的断言入口，保留它们真实的请求结构。

    参数:
        request: 离线客户端捕获的请求字典。
        provider: openai 或 anthropic。

    返回:
        系统提示词文本 (str)。

    行为说明:
        1. Anthropic 从顶层 system 读取，OpenAI 从 system 消息读取。
    """
    if provider == "anthropic":
        return request["system"]
    return next(message["content"] for message in request["messages"] if message["role"] == "system")


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
def test_turn_time_survives_midnight_tool_rounds_and_refreshes_next_input(tmp_path, local_clock, provider, stream):
    """验证跨午夜的多次工具续轮共用时间，下一用户回合重新取时。

    作用:
        从 respond 到实际协议发参验证时间注入，防止每个模型步骤重新读取时钟。

    参数:
        tmp_path: pytest 提供的隔离数据库目录。
        local_clock: 可推进的本机时钟与本地时区。
        provider: 实际使用的模型协议。
        stream: 是否走流式模型请求路径。

    返回:
        无返回值，通过请求、取时次数和历史断言验证行为。

    行为说明:
        1. 在午夜前开始回合，两次工具执行分别将本机时间推进一分钟。
        2. 检查三次请求共用原时间，下一用户输入改用午夜后的时间。
        3. 检查动态信息只进入请求 system，原始历史及落库消息保持原文。
    """
    model, requests = offline_model(provider, ["advance_1", "advance_2", None, None])
    before = f"2026-10-20T23:59:30{local_clock.offset}"
    after = f"2026-10-21T00:01:30{local_clock.offset}"

    def advance_clock():
        """推进本机时钟以模拟工具耗时跨过午夜。

        作用:
            在真实工具执行路径中改变后续取时结果，用于识别错误的逐请求刷新。

        返回:
            不含时间信息的工具结果 (dict)。

        行为说明:
            1. 只推进离线时钟，不修改系统时间或访问外部数据。
        """
        local_clock.value += timedelta(minutes=1)
        return {"ok": True}

    with create_runtime(
        model=model, db_path=tmp_path / "time.sqlite3", system_prompt="原有系统规则", auto_memory=False
    ) as runtime:
        runtime.tool_registry.handlers["advance_clock"] = advance_clock
        visible = []
        first = runtime.respond("查今天的历史", on_event=visible.append if stream else None)
        assert len(first.tool_results) == 2
        assert local_clock.now.call_count == 1
        second = runtime.respond("现在是哪天", on_event=visible.append if stream else None)
        assert local_clock.now.call_count == 2
        raw = runtime.history()
        persisted = [message["payload"] for message in runtime.session_store.inspect(runtime.session_id)["messages"]]

    systems = [system_text(request, provider) for request in requests]
    assert len(systems) == 4
    assert systems[0] == systems[1] == systems[2]
    assert before in systems[0] and "星期二" in systems[0]
    assert after in systems[3] and "星期三" in systems[3]
    assert before not in systems[3]
    for prompt in systems:
        assert prompt.startswith("原有系统规则")
        assert prompt.count("[本轮时间参考]") == 1
        assert local_clock.name in prompt
        assert "同一轮工具调用" in prompt
        assert "历史消息" in prompt
    assert [bool(request.get("stream")) for request in requests] == [stream] * 4
    conversation = [message for request in requests for message in request["messages"] if message["role"] != "system"]
    assert "[本轮时间参考]" not in json.dumps(conversation, ensure_ascii=False)
    assert raw == persisted == second.messages
    assert raw[0] == {"role": "system", "content": "原有系统规则"}
    assert [message["content"] for message in raw if message["role"] == "user"] == ["查今天的历史", "现在是哪天"]
    assert "[本轮时间参考]" not in json.dumps([raw, first.messages], ensure_ascii=False)
    if stream:
        assert [event["text"] for event in visible if event["event"] == "text_delta"] == ["已完成", "已完成"]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_restored_session_gets_current_time_without_rewriting_original_history(tmp_path, local_clock, provider):
    """验证恢复旧会话后使用新时间且不污染已保存历史。

    作用:
        防止时间被固化在创建会话时的系统消息中，或恢复时重复注入旧参考。

    参数:
        tmp_path: pytest 提供的隔离数据库目录。
        local_clock: 可推进的本机时钟与本地时区。
        provider: 实际使用的模型协议。

    返回:
        无返回值，通过恢复请求及原始消息记录断言验证行为。

    行为说明:
        1. 完成一次旧会话后关闭运行时，并将本机日期推进七天。
        2. 从同一临时数据库恢复并发出新输入，检查请求中的时间已更新。
        3. 保留既有消息内容和 ID，不将任何时间参考作为原始消息保存。
    """
    database = tmp_path / "restored.sqlite3"
    model, old_requests = offline_model(provider, [None])
    with create_runtime(model=model, db_path=database, system_prompt="旧会话规则", auto_memory=False) as runtime:
        runtime.respond("旧问题")
        session_id = runtime.session_id
        original = runtime.session_store.inspect(session_id)["messages"]
    old_prompt = system_text(old_requests[0], provider)
    local_clock.value += timedelta(days=7)
    model, new_requests = offline_model(provider, [None])
    with create_runtime(model=model, db_path=database, session_id=session_id, auto_memory=False) as restored:
        assert restored.session_store.inspect(session_id)["messages"] == original
        result = restored.respond("继续旧会话")
        records = restored.session_store.inspect(session_id)["messages"]
        assert records[: len(original)] == original

    assert local_clock.now.call_count == 2
    prompt = system_text(new_requests[0], provider)
    assert f"2026-10-20T23:59:30{local_clock.offset}" in old_prompt
    assert f"2026-10-27T23:59:30{local_clock.offset}" in prompt
    assert "2026-10-20T23:59:30" not in prompt
    assert prompt.count("[本轮时间参考]") == 1
    assert result.messages[: len(original)] == [record["payload"] for record in original]
    assert result.messages[-2] == {"role": "user", "content": "继续旧会话"}
    assert "[本轮时间参考]" not in json.dumps([record["payload"] for record in records], ensure_ascii=False)
