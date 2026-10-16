"""用临时 SQLite 写入故障验证工具执行与持久化之间的边界。"""

import json
import sqlite3

import pytest

from kairos.runtime import create_runtime


class Model:
    model = "recovery-test"

    def __init__(self, responses):
        self.responses = iter(responses)

    def complete(self, **kwargs):
        return next(self.responses)


def tool_response():
    return {
        "content": None,
        "tool_calls": [{"id": key, "function": {"name": "execute", "arguments": "{}"}} for key in ("first", "second")],
    }


def block_message_writes(connection, role):
    assert role in {"assistant", "tool"}
    connection.execute(
        f"""CREATE TEMP TRIGGER reject_message BEFORE INSERT ON session_messages
            WHEN NEW.role = '{role}' BEGIN SELECT RAISE(ABORT, '模拟保存失败'); END"""
    )


def test_failed_assistant_save_prevents_tool_execution_and_requires_explicit_resume(tmp_path):
    effects = []
    with create_runtime(
        model=Model([tool_response(), {"content": "恢复后继续"}]), db_path=tmp_path / "session.sqlite3"
    ) as runtime:
        runtime.tool_registry.handlers["execute"] = lambda: effects.append("effect") or {"ok": True}
        block_message_writes(runtime.connection, "assistant")
        with pytest.raises(sqlite3.IntegrityError, match="模拟保存失败"):
            runtime.respond("执行工具")
        assert effects == []
        assert not runtime.last_input_retained
        assert len(runtime.history()) == 1
        raw = runtime.session_store.inspect(runtime.session_id)
        assert raw["turns"][0]["status"] == "failed"
        assert not raw["messages"][1]["in_context"]
        assert not raw["tool_calls"]
        with pytest.raises(RuntimeError, match="保存曾失败"):
            runtime.respond("不能静默继续")

        runtime.connection.execute("DROP TRIGGER reject_message")
        runtime.resume_session(runtime.session_id)
        assert runtime.respond("继续").content == "恢复后继续"
        assert effects == []


def test_failed_tool_result_save_recovers_unknown_and_unexecuted_results_without_replay(tmp_path):
    database = tmp_path / "session.sqlite3"
    effects = []
    with create_runtime(model=Model([tool_response()]), db_path=database) as runtime:
        session_id = runtime.session_id
        runtime.tool_registry.handlers["execute"] = lambda: effects.append("effect") or {"ok": True}
        block_message_writes(runtime.connection, "tool")
        with pytest.raises(sqlite3.IntegrityError):
            runtime.respond("执行两个工具")
        assert runtime.last_input_retained
        assert effects == ["effect"]
        raw = runtime.session_store.inspect(session_id)
        assert [call["status"] for call in raw["tool_calls"]] == ["running", "pending"]
        assert all(call["result_message_id"] is None for call in raw["tool_calls"])

    with create_runtime(
        model=Model([{"content": "先核实执行结果"}]), db_path=database, session_id=session_id
    ) as restored:
        assert restored.recovered_turns == 1
        assert effects == ["effect"]
        results = [message for message in restored.messages if message["role"] == "tool"]
        assert [result["tool_call_id"] for result in results] == ["first", "second"]
        assert all(set(result) == {"role", "tool_call_id", "content"} for result in results)
        assert "尚未确认" in json.loads(results[0]["content"])["error"]["message"]
        assert "尚未开始" in json.loads(results[1]["content"])["error"]["message"]
        restored.respond("继续核实")
        assert effects == ["effect"]
        assert restored.session_store.recover_session(session_id) == 0


def test_reentrant_input_is_rejected_without_resetting_outer_turn_state(tmp_path):
    with create_runtime(model=Model([{"content": "完整回答"}]), db_path=tmp_path / "session.sqlite3") as runtime:

        def observe(event):
            if event["event"] == "model_request":
                with pytest.raises(RuntimeError, match="仍在执行"):
                    runtime.respond("回调中的重复输入")
                assert runtime.last_input_retained

        result = runtime.respond("原输入", on_event=observe)
        assert result.content == "完整回答"
        assert runtime.last_input_retained
        assert len(runtime.session_details()["turns"]) == 1


def test_database_tool_binding_error_rolls_back_before_persisting_error_result(tmp_path):
    response = {
        "tool_calls": [
            {
                "id": "bad_binding",
                "function": {
                    "name": "add_todo",
                    "arguments": json.dumps({"title": "测试事务", "due_at": {"invalid": "object"}}),
                },
            }
        ]
    }
    with create_runtime(
        model=Model([response, {"content": "参数有误，请补充日期"}, {"content": "可以继续对话"}]),
        db_path=tmp_path / "session.sqlite3",
    ) as runtime:
        first = runtime.respond("尝试添加")
        assert first.tool_results[0].value["error"]["type"] == "ProgrammingError"
        assert not runtime.connection.in_transaction
        assert runtime.session_details()["tool_calls"][0]["status"] == "failed"
        assert runtime.respond("继续").content == "可以继续对话"
        runtime.resume_session(runtime.session_id)
        assert runtime.recovered_turns == 0
