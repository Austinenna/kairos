"""会话存储的持久化、关联和中断恢复测试，全部使用临时数据库。"""

import json
import sqlite3

import pytest

from kairos.sessions import SessionStore


@pytest.fixture
def connection(tmp_path):
    connection = sqlite3.connect(tmp_path / "sessions.sqlite3")
    yield connection
    connection.close()


@pytest.fixture
def store(connection):
    return SessionStore(connection)


def new_session(store, **kwargs):
    return store.create_session("系统提示词", provider="openai", model="fake-model", **kwargs)


def call(call_id="call_1", *, arguments='{"query":"资料"}'):
    return {"id": call_id, "type": "function", "function": {"name": "search", "arguments": arguments}}


def user(text="检索资料"):
    return {"role": "user", "content": text}


def assistant(*calls):
    return {"role": "assistant", "content": "正在检索", "tool_calls": list(calls)}


def tool(call_id="call_1", content='{"items":[]}'):
    return {"role": "tool", "tool_call_id": call_id, "name": "search", "content": content}


def test_ids_relationships_ordering_and_no_open_transactions(store, connection):
    session = new_session(store)
    assert connection.row_factory is None
    assert not connection.in_transaction
    turn = store.begin_turn(session.session_id, user())
    step = store.start_step(turn.turn_id, 1, "fake-model")
    request = store.append_assistant(step.step_id, assistant(call("c1"), call("c2")))
    store.finish_step(step.step_id, "running", usage={"prompt_tokens": 20}, duration_ms=1.2)
    pending_step = store.inspect(session.session_id)["steps"][0]
    assert pending_step["status"] == "running"
    assert pending_step["usage"] == {"prompt_tokens": 20}
    assert pending_step["finished_at"] is None
    store.start_tool(step.step_id, "c1")
    first_result = store.append_tool_result(step.step_id, tool("c1"))
    store.start_tool(step.step_id, "c2")
    second_result = store.append_tool_result(step.step_id, tool("c2"))
    store.finish_step(step.step_id, "completed", usage={"prompt_tokens": 20}, duration_ms=1.2)
    final_step = store.start_step(turn.turn_id, 2, "fake-model")
    store.append_assistant(final_step.step_id, {"role": "assistant", "content": "整理好了"})
    store.finish_step(final_step.step_id, "completed", finish_reason="stop")
    store.finish_turn(turn.turn_id, "completed", "stop")

    records = store.load_messages(session.session_id)
    assert [record.seq for record in records] == list(range(6))
    assert [record.role for record in records] == ["system", "user", "assistant", "tool", "tool", "assistant"]
    assert len({record.message_id for record in records}) == 6
    assert records[0].turn_id is None and records[0].step_id is None
    assert records[1].message_id == turn.user_message_id
    assert records[1].turn_id == turn.turn_id and records[1].step_id is None
    assert {record.step_id for record in records[2:5]} == {step.step_id}
    assert records[-1].step_id == final_step.step_id
    assert all(record.session_id == session.session_id for record in records)

    detail = store.inspect(session.session_id)
    calls = detail["tool_calls"]
    assert [entry["assistant_message_id"] for entry in calls] == [request.message_id, request.message_id]
    assert [entry["result_message_id"] for entry in calls] == [first_result.message_id, second_result.message_id]
    assert detail["steps"][0]["usage"] == {"prompt_tokens": 20}
    assert detail["steps"][0]["finished_at"] is not None
    assert detail["session"]["message_count"] == 5
    assert detail["turns"][0]["status"] == "completed"
    assert detail["messages"][3]["tool_call_id"] == "c1"
    json.dumps(detail, ensure_ascii=False)
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert not connection.in_transaction


def test_cross_connection_restore_preserves_images_and_provider_fields(tmp_path):
    path = tmp_path / "roundtrip.sqlite3"
    connection = sqlite3.connect(path)
    store = SessionStore(connection)
    session = new_session(store, thinking_enabled=True)
    image = {"type": "image", "media_type": "image/png", "data": "aGVsbG8=", "name": "测试图.png"}
    input_payload = {"role": "user", "content": [{"type": "text", "text": "看图"}, image]}
    turn = store.begin_turn(session.session_id, input_payload)
    step = store.start_step(turn.turn_id, 1, "fake-model")
    output_payload = {
        "role": "assistant",
        "content": "看到了",
        "reasoning_content": "完整思考",
        "provider_content": [
            {"type": "thinking", "thinking": "完整思考", "signature": "provider-signature"},
            {"type": "text", "text": "看到了"},
        ],
        "id": "provider-message-id",
    }
    response = store.append_assistant(step.step_id, output_payload)
    store.finish_step(step.step_id, "completed")
    store.finish_turn(turn.turn_id, "completed", "stop")
    ids = [record.message_id for record in store.load_messages(session.session_id)]
    connection.close()

    with sqlite3.connect(path) as reopened:
        restored = SessionStore(reopened)
        records = restored.load_messages(session.session_id)
        assert [record.message_id for record in records] == ids
        assert records[1].payload == input_payload
        assert records[2].payload == output_payload
        assert response.message_id != output_payload["id"]
        assert restored.get_session(session.session_id).thinking_enabled is True
        assert restored.recover_session(session.session_id) == 0
    reopened.close()


def test_tool_call_id_is_scoped_to_step_and_original_value_is_preserved(store):
    session = new_session(store)
    turn = store.begin_turn(session.session_id, user())
    for index in (1, 2):
        step = store.start_step(turn.turn_id, index, "fake-model")
        store.append_assistant(step.step_id, assistant(call(" same_id ")))
        store.start_tool(step.step_id, " same_id ")
        store.append_tool_result(step.step_id, tool(" same_id "))
        store.finish_step(step.step_id, "completed")
    store.finish_turn(turn.turn_id, "completed", "stop")
    calls = store.inspect(session.session_id)["tool_calls"]
    assert [entry["tool_call_id"] for entry in calls] == [" same_id ", " same_id "]
    assert calls[0]["step_id"] != calls[1]["step_id"]
    assert calls[0]["result_message_id"] != calls[1]["result_message_id"]


def test_session_names_empty_listing_and_activity_order(store):
    empty = new_session(store)
    assert store.list_sessions() == []
    with pytest.raises(ValueError, match="历史会话"):
        store.latest_session()
    named = new_session(store, title="手动标题")
    assert [item.session_id for item in store.list_sessions()] == [named.session_id]
    turn = store.begin_turn(empty.session_id, user("第一行问题\n补充条件"))
    store.finish_turn(turn.turn_id, "failed", "model_error", retain_input=False)
    assert store.get_session(empty.session_id).title == "第一行问题"
    assert store.latest_session().session_id == empty.session_id
    assert store.get_session(empty.session_id).message_count == 0
    next_turn = store.begin_turn(empty.session_id, user("第二次输入"))
    store.finish_turn(next_turn.turn_id, "failed", "model_error", retain_input=False)
    assert store.get_session(empty.session_id).title == "第一行问题"

    store.begin_turn(named.session_id, user("不覆盖手动标题"))
    assert store.get_session(named.session_id).title == "手动标题"
    store.rename_session(empty.session_id, "重命名")
    assert store.latest_session().session_id == empty.session_id
    assert store.latest_session().title == "重命名"
    store.update_thinking(named.session_id, True)
    assert store.latest_session().thinking_enabled
    store.touch_session(empty.session_id)
    assert store.latest_session().session_id == empty.session_id
    assert len(store.list_sessions(limit=1)) == 1
    assert store.get_session(named.session_id).created_at.endswith("+00:00")


def test_reference_supports_unique_prefix_and_rejects_ambiguity(store, monkeypatch):
    ids = iter(["abc-session-one", "message-one", "abc-session-two", "message-two"])
    monkeypatch.setattr("kairos.sessions._id", lambda: next(ids))
    first = new_session(store)
    second = new_session(store)
    assert store.get_session("abc-session-o").session_id == first.session_id
    assert store.get_session(second.session_id).session_id == second.session_id
    with pytest.raises(ValueError, match="匹配多个"):
        store.get_session("abc")
    with pytest.raises(ValueError, match="找不到"):
        store.get_session("%")
    with pytest.raises(ValueError, match="不能为空"):
        store.get_session(" ")


@pytest.mark.parametrize("calls", [(call("c1"), call("c1")), (call(""),), (call("  "),)])
def test_invalid_call_ids_never_save_half_assistant_batch(store, connection, calls):
    session = new_session(store)
    turn = store.begin_turn(session.session_id, user())
    step = store.start_step(turn.turn_id, 1, "fake-model")
    with pytest.raises(ValueError, match="工具调用 ID"):
        store.append_assistant(step.step_id, assistant(*calls))
    detail = store.inspect(session.session_id)
    assert [record["role"] for record in detail["messages"]] == ["system", "user"]
    assert detail["tool_calls"] == []
    assert not connection.in_transaction


def test_database_failure_rolls_back_entire_assistant_batch(store, connection):
    session = new_session(store)
    turn = store.begin_turn(session.session_id, user())
    step = store.start_step(turn.turn_id, 1, "fake-model")
    connection.executescript("""
        CREATE TEMP TRIGGER fail_second_call BEFORE INSERT ON session_tool_calls
        WHEN NEW.tool_call_id = 'c2' BEGIN SELECT RAISE(ABORT, '模拟数据库失败'); END;
    """)
    with pytest.raises(sqlite3.IntegrityError, match="模拟数据库失败"):
        store.append_assistant(step.step_id, assistant(call("c1"), call("c2")))
    assert store.inspect(session.session_id)["tool_calls"] == []
    assert len(store.load_messages(session.session_id)) == 2
    assert not connection.in_transaction
    connection.execute("DROP TRIGGER fail_second_call")
    store.append_assistant(step.step_id, assistant(call("c1")))
    assert len(store.inspect(session.session_id)["tool_calls"]) == 1


def test_tool_result_and_call_link_are_atomic(store, connection):
    session = new_session(store)
    turn = store.begin_turn(session.session_id, user())
    step = store.start_step(turn.turn_id, 1, "fake-model")
    store.append_assistant(step.step_id, assistant(call()))
    connection.executescript("""
        CREATE TEMP TRIGGER fail_result_link BEFORE UPDATE ON session_tool_calls
        WHEN NEW.result_message_id IS NOT NULL BEGIN SELECT RAISE(ABORT, '模拟结果关联失败'); END;
    """)
    with pytest.raises(sqlite3.IntegrityError, match="模拟结果关联失败"):
        store.append_tool_result(step.step_id, tool())
    detail = store.inspect(session.session_id)
    assert len(detail["messages"]) == 3
    assert detail["tool_calls"][0]["result_message_id"] is None
    assert not connection.in_transaction


def test_failed_input_remains_auditable_but_leaves_model_context(store):
    session = new_session(store)
    turn = store.begin_turn(session.session_id, user())
    step = store.start_step(turn.turn_id, 1, "fake-model")
    store.finish_step(step.step_id, "failed", error={"type": "ModelError"})
    store.finish_turn(turn.turn_id, "failed", "model_error", retain_input=False, error="请求失败")
    assert [record.role for record in store.load_messages(session.session_id)] == ["system"]
    records = store.load_messages(session.session_id, include_inactive=True)
    assert records[1].message_id == turn.user_message_id
    assert records[1].payload == user()
    assert records[1].in_context is False
    detail = store.inspect(session.session_id)
    assert detail["turns"][0]["error"] == "请求失败"
    assert detail["steps"][0]["error"] == {"type": "ModelError"}
    assert store.recover_session(session.session_id) == 0


def test_cannot_remove_input_after_saved_assistant_or_duplicate_result(store):
    session = new_session(store)
    turn = store.begin_turn(session.session_id, user())
    step = store.start_step(turn.turn_id, 1, "fake-model")
    store.append_assistant(step.step_id, assistant(call()))
    with pytest.raises(ValueError, match="不能撤回"):
        store.finish_turn(turn.turn_id, "failed", "model_error", retain_input=False)
    assert store.inspect(session.session_id)["turns"][0]["status"] == "running"
    store.start_tool(step.step_id, "call_1")
    with pytest.raises(ValueError, match="不能重复执行"):
        store.start_tool(step.step_id, "call_1")
    result = store.append_tool_result(step.step_id, tool(content='{"error":{"message":"搜索失败"}}'))
    assert store.inspect(session.session_id)["tool_calls"][0]["status"] == "failed"
    with pytest.raises(ValueError, match="不能重复追加"):
        store.append_tool_result(step.step_id, tool())
    assert store.inspect(session.session_id)["tool_calls"][0]["result_message_id"] == result.message_id


def test_recover_persisted_calls_without_replaying_or_changing_completed_results(tmp_path):
    path = tmp_path / "interrupted.sqlite3"
    connection = sqlite3.connect(path)
    store = SessionStore(connection)
    session = new_session(store)
    turn = store.begin_turn(session.session_id, user())
    step = store.start_step(turn.turn_id, 1, "fake-model")
    store.append_assistant(step.step_id, assistant(call("done"), call("unknown"), call("pending")))
    store.start_tool(step.step_id, "done")
    done = store.append_tool_result(step.step_id, tool("done", '{"created":true}'))
    store.start_tool(step.step_id, "unknown")
    connection.close()

    connection = sqlite3.connect(path)
    try:
        recovered = SessionStore(connection)
        # 仅打开连接不会修改历史；用户显式恢复时才修补。
        assert len(recovered.load_messages(session.session_id)) == 4
        assert recovered.recover_session(session.session_id) == 1
        detail = recovered.inspect(session.session_id)
        assert [entry["status"] for entry in detail["tool_calls"]] == ["completed", "interrupted", "not_executed"]
        assert detail["tool_calls"][0]["result_message_id"] == done.message_id
        assert detail["messages"][3]["payload"]["content"] == '{"created":true}'
        assert "先查询实际状态" in detail["messages"][4]["payload"]["content"]
        assert "尚未开始执行" in detail["messages"][5]["payload"]["content"]
        assert all(set(record["payload"]) == {"role", "tool_call_id", "content"} for record in detail["messages"][4:6])
        assert detail["turns"][0]["status"] == "interrupted"
        assert detail["steps"][0]["status"] == "interrupted"
        assert recovered.recover_session(session.session_id) == 0
        assert len(recovered.load_messages(session.session_id)) == 6
        recovered.begin_turn(session.session_id, user("继续"))
        assert recovered.load_messages(session.session_id)[-1].seq == 6
        assert not connection.in_transaction
    finally:
        connection.close()


def test_recover_without_assistant_hides_input_and_recovers_unfinished_step(store):
    session = new_session(store)
    turn = store.begin_turn(session.session_id, user())
    store.start_step(turn.turn_id, 1, "fake-model")
    assert store.recover_session(session.session_id) == 1
    detail = store.inspect(session.session_id)
    assert detail["turns"][0]["status"] == "interrupted"
    assert detail["steps"][0]["status"] == "interrupted"
    assert detail["messages"][1]["in_context"] is False
    assert [record.role for record in store.load_messages(session.session_id)] == ["system"]
    assert store.recover_session(session.session_id) == 0


def test_recover_fills_missing_result_in_already_failed_turn(store):
    session = new_session(store)
    turn = store.begin_turn(session.session_id, user())
    step = store.start_step(turn.turn_id, 1, "fake-model")
    store.append_assistant(step.step_id, assistant(call()))
    store.finish_step(step.step_id, "failed")
    store.finish_turn(turn.turn_id, "failed", "model_error")
    assert store.recover_session(session.session_id) == 1
    detail = store.inspect(session.session_id)
    assert detail["turns"][0]["status"] == "failed"
    assert detail["steps"][0]["status"] == "failed"
    assert detail["tool_calls"][0]["status"] == "not_executed"
    assert store.recover_session(session.session_id) == 0


def test_foreign_keys_and_unrelated_tables_remain_intact(store, connection):
    connection.execute("CREATE TABLE unrelated (value TEXT)")
    connection.execute("INSERT INTO unrelated VALUES ('保留')")
    connection.commit()
    session = new_session(store)
    turn = store.begin_turn(session.session_id, user())
    with pytest.raises(ValueError, match="未结束"):
        store.begin_turn(session.session_id, user("不能同时开始"))
    step = store.start_step(turn.turn_id, 1, "fake-model")
    with pytest.raises(sqlite3.IntegrityError):
        store.start_step(turn.turn_id, 1, "fake-model")
    with pytest.raises(ValueError, match="找不到工具调用"):
        store.append_tool_result(step.step_id, tool("not-exist"))
    assert len(store.inspect(session.session_id)["messages"]) == 2
    assert connection.execute("SELECT value FROM unrelated").fetchone() == ("保留",)
    assert not connection.in_transaction
