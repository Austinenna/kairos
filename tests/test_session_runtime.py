"""会话持久化与现有模型、图片、工具边界的离线集成验证。"""

import copy
import json
from types import SimpleNamespace

import pytest
from PIL import Image

import kairos.runtime as runtime_module
from kairos.config import Settings
from kairos.context import REFERENCE_HEADER
from kairos.images import load_image
from kairos.providers import AnthropicMessagesModel, OpenAICompatibleModel
from kairos.runtime import create_runtime
from kairos_todo import list_todos


class FakeModel:
    model = "session-test-model"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, *, messages, tools):
        self.requests.append(copy.deepcopy(messages))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return copy.deepcopy(response)


def call(call_id, name, **arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def image_blocks(messages):
    return [
        block
        for message in messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "image"
    ]


def conversation_messages(messages):
    """检查请求只有一份记忆参考，并取出原始对话部分。

    作用:
        让会话恢复断言聚焦真实用户、助手和工具记录，同时验证新增记忆参考没有重复注入。

    参数:
        messages: ContextBuilder 或协议适配器处理后的模型请求消息。

    返回:
        去除 system 和单条合成记忆参考后的对话消息 (list[dict])。

    行为说明:
        1. 通过固定参考标记检查合成消息恰好一条。
        2. 仅过滤合成参考与系统规则，不改写原始消息或跳过工具结果。
    """
    references = [message for message in messages if str(message.get("content", "")).startswith(REFERENCE_HEADER)]
    assert len(references) == 1
    return [message for message in messages if message["role"] != "system" and message not in references]


def test_session_switching_and_restart_keep_history_and_ids_separate(tmp_path):
    database = tmp_path / "sessions.sqlite3"
    first_model = FakeModel([{"content": "A 的回答"}, {"content": "B 的回答"}, {"content": "继续 A"}])
    with create_runtime(model=first_model, db_path=database, system_prompt="会话 A 的规则") as runtime:
        session_a = runtime.session_id
        first_turn = runtime.respond("A 的问题")
        before = runtime.session_store.inspect(session_a)
        session_b = runtime.new_session(title="会话 B", system_prompt="会话 B 的规则").session_id
        assert session_a != session_b
        runtime.respond("B 的问题")
        assert first_model.requests[1][0]["content"].startswith("会话 B 的规则\n\n")
        assert conversation_messages(first_model.requests[1]) == [
            {"role": "user", "content": "B 的问题"},
        ]
        runtime.resume_session(session_a)
        second_turn = runtime.respond("继续 A 的问题")
        assert second_turn.turn_id != first_turn.turn_id
        assert second_turn.session_id == session_a
        assert all("B 的" not in str(message) for message in first_model.requests[-1])
        old_ids = [message["message_id"] for message in before["messages"]]
        persisted = runtime.session_store.inspect(session_a)
        assert [message["message_id"] for message in persisted["messages"][: len(old_ids)]] == old_ids
        assert {session_a, session_b} <= {session.session_id for session in runtime.list_sessions()}

    restored_model = FakeModel([{"content": "恢复后继续 B"}])
    with create_runtime(model=restored_model, db_path=database, session_id=session_b) as restored:
        assert restored.session_id == session_b
        restored.respond("B 的后续问题")
        assert restored_model.requests[0][0]["content"].startswith("会话 B 的规则\n\n")
        assert conversation_messages(restored_model.requests[0]) == [
            {"role": "user", "content": "B 的问题"},
            {"role": "assistant", "content": "B 的回答"},
            {"role": "user", "content": "B 的后续问题"},
        ]
        restored.rename_session("重命名后的 B")
        assert restored.session_info.title == "重命名后的 B"

    with create_runtime(model=FakeModel([]), db_path=database, resume_latest=True) as latest:
        assert latest.session_id == session_b
        assert latest.session_info.title == "重命名后的 B"


def test_five_ids_link_multiple_calls_steps_and_turns_without_leaking_into_model_messages(tmp_path):
    responses = [
        {
            "id": "provider-response",
            "tool_calls": [call("shared", "lookup", item="A"), call("other", "lookup", item="B")],
        },
        {"id": "provider-response", "tool_calls": [call("shared", "lookup", item="C")]},
        {"id": "provider-response", "content": "第一轮结束"},
        {"id": "provider-response", "tool_calls": [call("shared", "lookup", item="D")]},
        {"id": "provider-response", "content": "第二轮结束"},
    ]
    model = FakeModel(responses)
    with create_runtime(model=model, db_path=tmp_path / "ids.sqlite3") as runtime:
        runtime.tool_registry.handlers["lookup"] = lambda item: {"found": item}
        first = runtime.respond("查找 A、B、C")
        second = runtime.respond("再查找 D")
        details = runtime.session_store.inspect(runtime.session_id)

    assert first.session_id == second.session_id == details["session"]["session_id"]
    turns = {turn["turn_id"]: turn for turn in details["turns"]}
    assert set(turns) == {first.turn_id, second.turn_id}
    assert all(turn["session_id"] == first.session_id and turn["status"] == "completed" for turn in turns.values())
    steps = {step["step_id"]: step for step in details["steps"]}
    assert len(steps) == 5
    assert [steps[step_id]["step_index"] for step_id in first.step_ids] == [1, 2, 3]
    assert [steps[step_id]["step_index"] for step_id in second.step_ids] == [1, 2]
    assert all(steps[step_id]["turn_id"] == first.turn_id for step_id in first.step_ids)
    assert all(steps[step_id]["turn_id"] == second.turn_id for step_id in second.step_ids)

    messages = {message["message_id"]: message for message in details["messages"]}
    assert len(messages) == len(details["messages"])
    assert "provider-response" not in messages
    assert [message["seq"] for message in details["messages"]] == list(range(len(messages)))
    system = details["messages"][0]
    assert system["role"] == "system" and system["turn_id"] is None and system["step_id"] is None
    for turn in turns.values():
        user = messages[turn["user_message_id"]]
        assert user["role"] == "user" and user["turn_id"] == turn["turn_id"] and user["step_id"] is None
    for message in messages.values():
        assert message["session_id"] == first.session_id
        if message["step_id"] is not None:
            assert steps[message["step_id"]]["turn_id"] == message["turn_id"]

    calls = details["tool_calls"]
    assert len(calls) == 4
    assert len({(tool["step_id"], tool["tool_call_id"]) for tool in calls}) == 4
    assert sum(tool["tool_call_id"] == "shared" for tool in calls) == 3
    found = []
    for tool in calls:
        assistant = messages[tool["assistant_message_id"]]
        result = messages[tool["result_message_id"]]
        assert assistant["role"] == "assistant" and result["role"] == "tool"
        assert assistant["step_id"] == result["step_id"] == tool["step_id"]
        assert result["tool_call_id"] == tool["tool_call_id"]
        assert any(item["id"] == tool["tool_call_id"] for item in assistant["payload"]["tool_calls"])
        found.append(json.loads(result["payload"]["content"])["found"])
    assert found == ["A", "B", "C", "D"]
    assert first.tool_results[0].step_id == first.tool_results[1].step_id == first.step_ids[0]
    assert first.tool_results[2].step_id == first.step_ids[1]
    assert second.tool_results[0].step_id == second.step_ids[0]

    metadata = {"session_id", "turn_id", "step_id", "message_id", "seq", "in_context"}
    for payload in [*second.messages, *(message for request in model.requests for message in request)]:
        assert not metadata.intersection(payload)
    assert [message["payload"] for message in details["messages"] if message["in_context"]] == second.messages


def test_image_snapshot_and_observation_ids_survive_restart_without_exposing_image_data(tmp_path):
    source = tmp_path / "snapshot.png"
    Image.new("RGB", (40, 30), (12, 34, 56)).save(source)
    picture = load_image(source)
    database = tmp_path / "images.sqlite3"
    log_path = tmp_path / "events.jsonl"
    visible = []
    with create_runtime(model=FakeModel([{"content": "看到了原图"}]), db_path=database, log_path=log_path) as runtime:
        result = runtime.respond("解释图片", images=[picture], on_event=visible.append)
        session_id = runtime.session_id
        raw = runtime.session_store.inspect(session_id)
        original_message_ids = [record["message_id"] for record in raw["messages"]]
        assert image_blocks([record["payload"] for record in raw["messages"]])[0]["data"] == picture.data
        assert picture.data not in json.dumps(runtime.session_details())
        assert picture.data not in json.dumps(runtime.history())
    source.write_bytes(b"original image no longer exists")

    model = FakeModel([{"content": "仍能继续分析原图"}])
    with create_runtime(model=model, db_path=database, session_id=session_id) as restored:
        before = restored.session_store.inspect(session_id)
        assert [record["message_id"] for record in before["messages"]] == original_message_ids
        restored.respond("原图中还有什么")
        assert image_blocks(model.requests[0])[0]["data"] == picture.data

    logs = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert any(event["event"] == "text_delta" for event in visible)
    for event in [*result.events, *visible, *logs]:
        assert event["session_id"] == session_id
        assert event["turn_id"] == result.turn_id
        assert event["step_id"] in result.step_ids
    assert picture.data not in json.dumps([result.events, visible, logs])


def test_anthropic_signed_block_order_is_preserved_in_restored_provider_request(tmp_path):
    blocks = [
        {"type": "thinking", "thinking": "先查询", "signature": "test-signature"},
        {"type": "text", "text": "准备查询。"},
        {"type": "tool_use", "id": "query_1", "name": "list_todos", "input": {}},
        {"type": "redacted_thinking", "data": "opaque-provider-data"},
    ]
    replies = iter(
        [
            {"content": blocks, "stop_reason": "tool_use"},
            {"content": [{"type": "text", "text": "没有待办。"}], "stop_reason": "end_turn"},
        ]
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **_: next(replies)))
    database = tmp_path / "anthropic.sqlite3"
    with create_runtime(model=AnthropicMessagesModel("test-model", client=client), db_path=database) as runtime:
        result = runtime.respond("查询待办")
        session_id = runtime.session_id
        assert result.messages[2]["provider_content"] == blocks
        assert result.messages[2]["reasoning_content"] == "先查询"

    requests = []

    def create(**request):
        requests.append(copy.deepcopy(request))
        return {"content": [{"type": "text", "text": "继续回答。"}], "stop_reason": "end_turn"}

    restored_client = SimpleNamespace(messages=SimpleNamespace(create=create))
    restored_model = AnthropicMessagesModel("test-model", client=restored_client)
    with create_runtime(model=restored_model, db_path=database, session_id=session_id) as restored:
        restored.respond("继续之前的对话")
    restored_history = conversation_messages(requests[0]["messages"])
    assistant = next(message for message in restored_history if message["role"] == "assistant")
    assert assistant == {"role": "assistant", "content": blocks}
    tool_result = next(
        message
        for message in restored_history
        if isinstance(message["content"], list) and message["content"][0].get("type") == "tool_result"
    )
    assert tool_result["content"] == [{"type": "tool_result", "tool_use_id": "query_1", "content": "[]"}]


@pytest.mark.parametrize("error,status", [(KeyboardInterrupt(), "cancelled"), (OSError("离线模拟错误"), "failed")])
def test_failed_input_is_kept_for_audit_but_excluded_from_restored_context(tmp_path, error, status):
    database = tmp_path / "failed.sqlite3"
    with create_runtime(model=FakeModel([error]), db_path=database) as runtime:
        session_id = runtime.session_id
        with pytest.raises(type(error)):
            runtime.respond("本轮未完成的输入")
        assert not runtime.last_input_retained
        details = runtime.session_store.inspect(session_id)
        turn = details["turns"][0]
        assert turn["status"] == status
        user = next(message for message in details["messages"] if message["message_id"] == turn["user_message_id"])
        assert user["payload"]["content"] == "本轮未完成的输入"
        assert not user["in_context"]
        assert len(runtime.history()) == 1

    model = FakeModel([{"content": "新的回答"}])
    with create_runtime(model=model, db_path=database, session_id=session_id) as restored:
        assert len(restored.history()) == 1
        completed = restored.respond("新的输入")
        assert completed.turn_id != turn["turn_id"]
        assert all(message.get("content") != "本轮未完成的输入" for message in model.requests[0])
        records = restored.session_store.inspect(session_id)["messages"]
        assert [record["seq"] for record in records] == list(range(len(records)))
        assert any(record["message_id"] == user["message_id"] and not record["in_context"] for record in records)


@pytest.mark.parametrize("error", [KeyboardInterrupt(), OSError("工具执行后的离线错误")])
def test_restoring_interrupted_turn_keeps_completed_tool_and_never_reexecutes_it(tmp_path, error):
    database = tmp_path / "interrupted.sqlite3"
    model = FakeModel([{"tool_calls": [call("add_1", "add_todo", title="只创建一次")]}, error])
    with create_runtime(model=model, db_path=database) as runtime:
        session_id = runtime.session_id
        with pytest.raises(type(error)):
            runtime.respond("添加一条待办")
        assert runtime.last_input_retained
        assert len(list_todos(runtime.connection)) == 1
        details = runtime.session_store.inspect(session_id)
        assert len(details["tool_calls"]) == 1
        tool = details["tool_calls"][0]
        result = next(message for message in details["messages"] if message["message_id"] == tool["result_message_id"])
        assert result["in_context"] and result["tool_call_id"] == "add_1"

    followup = FakeModel([{"content": "已根据历史确认，无需重复创建。"}])
    with create_runtime(model=followup, db_path=database, session_id=session_id) as restored:
        assert len(list_todos(restored.connection)) == 1
        restored.respond("刚才是否已经创建")
        assert len(list_todos(restored.connection)) == 1
        restored_history = conversation_messages(followup.requests[0])
        assert [message["role"] for message in restored_history] == ["user", "assistant", "tool", "user"]
        tool_result = next(message for message in restored_history if message["role"] == "tool")
        assert json.loads(tool_result["content"])["title"] == "只创建一次"
        assert tool_result["tool_call_id"] == "add_1"
        assert restored.session_store.inspect(session_id)["tool_calls"] == details["tool_calls"]


def test_reset_creates_new_session_and_restores_original_prompt_and_thinking(tmp_path):
    database = tmp_path / "settings.sqlite3"
    responses = iter(["旧会话回答", "新会话回答"])
    requests = []

    def create(**request):
        requests.append(copy.deepcopy(request))
        return {"choices": [{"message": {"content": next(responses)}, "finish_reason": "stop"}]}

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    model = OpenAICompatibleModel(model="deepseek-test", client=client)
    with create_runtime(model=model, db_path=database, system_prompt="原始规则", thinking_enabled=True) as runtime:
        old_id = runtime.session_id
        runtime.respond("旧会话问题")
        runtime.reset(system_prompt="新规则")
        new_id = runtime.session_id
        assert new_id != old_id
        assert runtime.history() == [{"role": "system", "content": "新规则"}]
        runtime.set_thinking(False)
        runtime.respond("新会话问题")
        runtime.resume_session(old_id)
        assert runtime.system_prompt == "原始规则"
        assert runtime.thinking_enabled
        assert runtime.history()[-1]["content"] == "旧会话回答"
        runtime.resume_session(new_id)
        assert runtime.system_prompt == "新规则"
        assert not runtime.thinking_enabled
        assert [request["extra_body"]["thinking"]["type"] for request in requests] == ["enabled", "disabled"]

    with create_runtime(
        model=OpenAICompatibleModel(model="deepseek-test", client=client), db_path=database, session_id=old_id
    ) as restored:
        assert restored.system_prompt == "原始规则"
        assert restored.thinking_enabled
        assert restored.model.extra_body["thinking"]["type"] == "enabled"
        assert restored.history()[-1]["content"] == "旧会话回答"


def test_resume_uses_saved_provider_and_model_but_current_connection_settings(tmp_path, monkeypatch):
    database = tmp_path / "provider-settings.sqlite3"
    original = Settings(
        _env_file=None,
        kairos_provider="anthropic",
        kairos_model="saved-anthropic-model",
        kairos_api_key="test-old-key",
        gateway_base_url="https://old-gateway.invalid/v1",
        anthropic_base_url="https://old-gateway.invalid",
    )
    client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **_: {"content": [{"type": "text", "text": "初始回答"}], "stop_reason": "end_turn"}
        )
    )
    with create_runtime(settings=original, client=client, db_path=database) as runtime:
        runtime.respond("保存一段 Anthropic 会话")
        session_id = runtime.session_id
        assert "test-old-key" not in json.dumps(runtime.session_store.inspect(session_id))

    current = Settings(
        _env_file=None,
        kairos_provider="openai",
        kairos_model="different-current-default",
        kairos_api_key="test-current-key",
        gateway_base_url="https://current-gateway.invalid/v1",
        anthropic_base_url="https://current-gateway.invalid",
    )
    requests = []
    configured = []
    original_build = runtime_module.build_model

    def build(settings, client=None, *, thinking_enabled=False):
        configured.append(settings)
        return original_build(settings, client=client, thinking_enabled=thinking_enabled)

    def create(**request):
        requests.append(request)
        return {"content": [{"type": "text", "text": "恢复成功"}], "stop_reason": "end_turn"}

    monkeypatch.setattr(runtime_module, "build_model", build)
    resumed_client = SimpleNamespace(messages=SimpleNamespace(create=create))
    with create_runtime(settings=current, client=resumed_client, db_path=database, session_id=session_id) as restored:
        assert isinstance(restored.model, AnthropicMessagesModel)
        assert restored.model_name == "saved-anthropic-model"
        restored.respond("继续旧会话")
        assert "test-current-key" not in json.dumps(restored.session_store.inspect(session_id))
    assert configured[-1].kairos_provider == "anthropic"
    assert configured[-1].kairos_model == "saved-anthropic-model"
    assert configured[-1].kairos_api_key == "test-current-key"
    assert configured[-1].gateway_base_url == "https://current-gateway.invalid/v1"
    assert configured[-1].anthropic_base_url == "https://current-gateway.invalid"
    assert requests[0]["model"] == "saved-anthropic-model"
