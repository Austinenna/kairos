import json
import sqlite3

import pytest

from kairos.pricing import Usage
from kairos.runtime import DEFAULT_SYSTEM_PROMPT, create_runtime
from kairos.todos import list_todos


class FakeModel:
    """按顺序返回预先准备的 Agent 响应，并记录每次请求。"""

    model = "fake-model"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, *, messages, tools):
        self.calls.append({"messages": [dict(message) for message in messages], "tools": list(tools)})
        return next(self.responses)


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
        assert [message["role"] for message in runtime.history()] == [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert model.calls[1]["messages"] == [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
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
        assert {"add_todo", "list_todos", "update_todo", "complete_todo"} == set(runtime.tool_registry.handlers)
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
