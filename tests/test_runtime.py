import json
import sqlite3

import pytest

from kairos.context import REFERENCE_HEADER
from kairos.pricing import Usage
from kairos.runtime import DEFAULT_SYSTEM_PROMPT, create_runtime
from kairos_todo import add_todo, list_todos


class FakeModel:
    """按顺序返回预先准备的 Agent 响应，并记录每次请求。"""

    model = "fake-model"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, *, messages, tools):
        self.calls.append({"messages": [dict(message) for message in messages], "tools": list(tools)})
        return next(self.responses)


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("delete_todo", {"todo_id": True}),
        ("complete_todo", {"todo_id": 1.0}),
        ("add_todo", {"title": "非法标签", "tags": "工作"}),
        ("update_todo", {"todo_id": 1, "due_at": "明天下午"}),
    ],
)
def test_invalid_model_arguments_return_error_without_mutating_todos(tmp_path, name, arguments):
    model = FakeModel(
        [
            {"tool_calls": [{"id": "invalid", "function": {"name": name, "arguments": json.dumps(arguments)}}]},
            {"content": "参数需要修正"},
        ]
    )
    with create_runtime(model=model, db_path=tmp_path / "kairos.sqlite3") as runtime:
        retained = add_todo(runtime.connection, "保留原记录")
        result = runtime.respond("模拟非法工具参数")
        assert result.tool_results[0].value["error"]["type"] == "ValueError"
        assert list_todos(runtime.connection) == [retained]
        assert runtime.session_details()["tool_calls"][0]["status"] == "failed"
        assert "error" in json.loads(model.calls[1]["messages"][-1]["content"])


def test_create_runtime_injects_fake_model_and_preserves_multi_turn_history(tmp_path):
    model = FakeModel(
        [
            {"content": "第一轮回答", "tool_calls": []},
            {"content": "第二轮回答", "tool_calls": []},
        ]
    )

    runtime = create_runtime(model=model, db_path=tmp_path / "kairos.sqlite3")
    try:
        first = runtime.respond("第一轮问题")
        second = runtime.respond("继续刚才的话题")

        assert first.content == "第一轮回答"
        assert second.content == "第二轮回答"
        assert runtime.model is model
        assert runtime.model_name == "fake-model"
        assert "调用对应工具" in runtime.history()[0]["content"]
        assert [message["role"] for message in runtime.history()] == [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        request = model.calls[1]["messages"]
        assert request[0]["content"].startswith(DEFAULT_SYSTEM_PROMPT + "\n\n")
        references = [message for message in request if str(message.get("content", "")).startswith(REFERENCE_HEADER)]
        assert len(references) == 1
        assert [message for message in request if message["role"] != "system" and message not in references] == [
            {"role": "user", "content": "第一轮问题"},
            {"role": "assistant", "content": "第一轮回答"},
            {"role": "user", "content": "继续刚才的话题"},
        ]
    finally:
        runtime.close()


def test_runtime_uses_registered_todo_tool_and_keeps_tool_messages(tmp_path):
    model = FakeModel(
        [
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_add",
                        "type": "function",
                        "function": {
                            "name": "add_todo",
                            "arguments": '{"title":"测试运行时工具"}',
                        },
                    }
                ],
            },
            {"content": "待办已创建。", "tool_calls": []},
        ]
    )

    runtime = create_runtime(model=model, db_path=tmp_path / "kairos.sqlite3")
    try:
        result = runtime.respond("创建一个测试待办")

        assert result.content == "待办已创建。"
        assert result.tool_results[0].name == "add_todo"
        assert result.tool_results[0].value["title"] == "测试运行时工具"
        assert {
            "add_todo",
            "list_todos",
            "update_todo",
            "complete_todo",
            "delete_todo",
            "remember",
            "read_memory",
            "search_history",
            "read_session",
        } == set(runtime.tool_registry.handlers)
        assert [todo["title"] for todo in list_todos(runtime.connection)] == ["测试运行时工具"]
        assert [message["role"] for message in result.messages] == [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert json.loads(result.messages[-2]["content"])["title"] == "测试运行时工具"
        assert len(model.calls) == 2
        assert model.calls[0]["tools"]
    finally:
        runtime.close()


def test_runtime_exposes_delete_schema_and_returns_deleted_record_to_model(tmp_path):
    model = FakeModel(
        [
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_delete",
                        "type": "function",
                        "function": {"name": "delete_todo", "arguments": '{"todo_id":1}'},
                    }
                ],
            },
            {"content": "待办已删除。", "tool_calls": []},
        ]
    )

    with create_runtime(model=model, db_path=tmp_path / "kairos.sqlite3") as runtime:
        target = add_todo(runtime.connection, "删除测试待办", tags=["测试"])
        retained = add_todo(runtime.connection, "保留另一条待办")

        result = runtime.respond(f"删除待办 {target['id']}")

        schema = next(tool["function"] for tool in model.calls[0]["tools"] if tool["function"]["name"] == "delete_todo")
        assert schema["parameters"]["required"] == ["todo_id"]
        assert schema["parameters"]["properties"]["todo_id"]["type"] == "integer"
        assert result.content == "待办已删除。"
        assert result.tool_results[0].name == "delete_todo"
        assert result.tool_results[0].call_id == "call_delete"
        assert result.tool_results[0].value == target
        assert list_todos(runtime.connection) == [retained]
        tool_message = model.calls[1]["messages"][-1]
        assert tool_message["role"] == "tool"
        assert tool_message["tool_call_id"] == "call_delete"
        assert json.loads(tool_message["content"]) == target


def test_runtime_returns_missing_delete_error_and_continues_session(tmp_path):
    model = FakeModel(
        [
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_delete_missing",
                        "type": "function",
                        "function": {"name": "delete_todo", "arguments": '{"todo_id":999}'},
                    }
                ],
            },
            {"content": "没有找到指定待办。", "tool_calls": []},
            {"content": "可以继续对话。", "tool_calls": []},
        ]
    )

    with create_runtime(model=model, db_path=tmp_path / "kairos.sqlite3") as runtime:
        retained = add_todo(runtime.connection, "保留已有待办")

        result = runtime.respond("删除待办 999")

        assert result.content == "没有找到指定待办。"
        error = result.tool_results[0].value["error"]
        assert error["type"] == "KeyError"
        assert "找不到待办：999" in error["message"]
        tool_message = model.calls[1]["messages"][-1]
        assert tool_message["role"] == "tool"
        assert tool_message["tool_call_id"] == "call_delete_missing"
        assert json.loads(tool_message["content"]) == {"error": error}
        assert list_todos(runtime.connection) == [retained]
        assert runtime.respond("继续对话").content == "可以继续对话。"
        assert tool_message in model.calls[2]["messages"]


def test_runtime_aggregates_usage_across_tool_rounds(tmp_path):
    model = FakeModel(
        [
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_list",
                        "type": "function",
                        "function": {"name": "list_todos", "arguments": "{}"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "cached_input_tokens": 3,
                    "completion_tokens": 4,
                    "reasoning_tokens": 1,
                },
            },
            {
                "content": "没有待办。",
                "tool_calls": [],
                "usage": {
                    "prompt_tokens": 20,
                    "cached_input_tokens": 2,
                    "completion_tokens": 5,
                    "reasoning_tokens": 0,
                },
            },
        ]
    )

    runtime = create_runtime(model=model, db_path=tmp_path / "kairos.sqlite3")
    try:
        result = runtime.respond("查询待办")

        assert result.usage == Usage(
            input_tokens=30,
            cached_input_tokens=5,
            output_tokens=9,
            reasoning_tokens=1,
        )
        assert [event["event"] for event in result.events] == [
            "model_request",
            "model_response",
            "tool_result",
            "model_request",
            "model_response",
        ]
    finally:
        runtime.close()


def test_runtime_reset_replaces_history_but_keeps_dependencies(tmp_path):
    model = FakeModel([{"content": "回答", "tool_calls": []}])
    runtime = create_runtime(model=model, db_path=tmp_path / "kairos.sqlite3")

    runtime.respond("保留之前的消息")
    connection = runtime.connection
    runtime.reset(system_prompt="新的系统提示词")

    assert runtime.connection is connection
    assert runtime.history() == [{"role": "system", "content": "新的系统提示词"}]
    assert runtime.last_result is None

    runtime.close()
    runtime.close()
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
    with pytest.raises(RuntimeError, match="已经关闭"):
        runtime.respond("关闭后的请求")
    with pytest.raises(RuntimeError, match="已经关闭"):
        runtime.reset()
