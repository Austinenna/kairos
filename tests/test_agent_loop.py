import json

import pytest

from kairos.agent_loop import AgentLoopLimitError, run_single_tool_turn


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, *, messages, tools):
        self.calls.append({"messages": list(messages), "tools": tools})
        return next(self.responses)


def test_single_tool_turn_executes_tool_and_returns_final_answer():
    model = FakeModel(
        [
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "add_todo",
                            "arguments": '{"title": "学习 Agent"}',
                        },
                    }
                ],
            },
            {"content": "已经帮你加好待办了。", "tool_calls": []},
        ]
    )
    called_arguments = []

    def add_todo(**arguments):
        called_arguments.append(arguments)
        return {"id": 1, "title": arguments["title"]}

    messages = [{"role": "user", "content": "把学习 Agent 加入待办"}]
    final_content, updated_messages = run_single_tool_turn(
        messages,
        model,
        {"add_todo": add_todo},
        [{"type": "function", "function": {"name": "add_todo"}}],
    )

    assert final_content == "已经帮你加好待办了。"
    assert called_arguments == [{"title": "学习 Agent"}]
    assert [message["role"] for message in updated_messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert updated_messages[2]["tool_call_id"] == "call_1"
    assert json.loads(updated_messages[2]["content"]) == {"id": 1, "title": "学习 Agent"}
    assert len(model.calls) == 2


def test_single_tool_turn_can_return_direct_model_answer():
    model = FakeModel([{"content": "今天没有需要处理的待办。", "tool_calls": []}])
    messages = [{"role": "user", "content": "今天有什么事？"}]

    final_content, updated_messages = run_single_tool_turn(messages, model, {}, [])

    assert final_content == "今天没有需要处理的待办。"
    assert updated_messages[-1] == {"role": "assistant", "content": "今天没有需要处理的待办。"}
    assert len(model.calls) == 1


def test_single_tool_turn_preserves_reasoning_content_in_assistant_message():
    model = FakeModel(
        [
            {
                "content": None,
                "reasoning_content": "需要先创建待办。",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "add_todo", "arguments": "{}"},
                    }
                ],
            },
            {"content": "完成。", "tool_calls": []},
        ]
    )

    _, updated_messages = run_single_tool_turn(
        [{"role": "user", "content": "加一个待办"}],
        model,
        {"add_todo": lambda: {"id": 1}},
        [],
    )

    assert updated_messages[1]["reasoning_content"] == "需要先创建待办。"


def test_single_tool_turn_executes_multiple_tool_calls_in_order():
    model = FakeModel(
        [
            {
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "first", "arguments": '{"value": 1}'}},
                    {"id": "call_2", "function": {"name": "second", "arguments": '{"value": 2}'}},
                ],
            },
            {"content": "两个操作都完成了。", "tool_calls": []},
        ]
    )
    called = []

    def handler(name):
        def run(**arguments):
            called.append((name, arguments))
            return {"name": name, "value": arguments["value"]}

        return run

    final_content, messages = run_single_tool_turn(
        [{"role": "user", "content": "执行两个操作"}],
        model,
        {"first": handler("first"), "second": handler("second")},
        [],
    )

    assert final_content == "两个操作都完成了。"
    assert called == [("first", {"value": 1}), ("second", {"value": 2})]
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "tool", "assistant"]


def test_single_tool_turn_continues_after_tool_result_until_final_answer():
    model = FakeModel(
        [
            {
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "first", "arguments": "{}"}},
                ],
            },
            {
                "content": None,
                "tool_calls": [
                    {"id": "call_2", "function": {"name": "second", "arguments": "{}"}},
                ],
            },
            {"content": "两步都完成了。", "tool_calls": []},
        ]
    )
    called = []

    final_content, messages = run_single_tool_turn(
        [{"role": "user", "content": "分两步执行"}],
        model,
        {
            "first": lambda: called.append("first") or {"ok": "first"},
            "second": lambda: called.append("second") or {"ok": "second"},
        },
        [],
    )

    assert final_content == "两步都完成了。"
    assert called == ["first", "second"]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]


def test_single_tool_turn_returns_invalid_json_as_tool_error():
    model = FakeModel(
        [
            {
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "demo", "arguments": "{not-json"}},
                ],
            },
            {"content": "参数有误，我没有执行这个工具。", "tool_calls": []},
        ]
    )

    answer, messages = run_single_tool_turn(
        [{"role": "user", "content": "测试非法参数"}],
        model,
        {"demo": lambda: {"ok": True}},
        [],
    )

    assert answer == "参数有误，我没有执行这个工具。"
    assert json.loads(messages[2]["content"]) == {
        "error": {
            "type": "JSONDecodeError",
            "message": "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)",
        }
    }
    assert messages[2]["tool_call_id"] == "call_1"
    assert len(model.calls) == 2


def test_single_tool_turn_returns_unknown_tool_as_tool_error():
    model = FakeModel(
        [
            {
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "missing_tool", "arguments": "{}"}},
                ],
            },
            {"content": "这个工具不存在。", "tool_calls": []},
        ]
    )

    answer, messages = run_single_tool_turn(
        [{"role": "user", "content": "测试未知工具"}],
        model,
        {},
        [],
    )

    assert answer == "这个工具不存在。"
    error = json.loads(messages[2]["content"])
    assert error["error"]["type"] == "KeyError"
    assert messages[2]["tool_call_id"] == "call_1"


def test_single_tool_turn_returns_handler_error_as_tool_error():
    def broken_handler():
        raise ValueError("演示错误")

    model = FakeModel(
        [
            {
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "demo", "arguments": "{}"}},
                ],
            },
            {"content": "工具执行失败，我已经知道原因。", "tool_calls": []},
        ]
    )

    answer, messages = run_single_tool_turn(
        [{"role": "user", "content": "测试工具异常"}],
        model,
        {"demo": broken_handler},
        [],
    )

    assert answer == "工具执行失败，我已经知道原因。"
    assert json.loads(messages[2]["content"]) == {"error": {"type": "ValueError", "message": "演示错误"}}


def test_single_tool_turn_continues_other_calls_after_one_tool_error():
    called = []
    model = FakeModel(
        [
            {
                "content": None,
                "tool_calls": [
                    {"id": "call_bad", "function": {"name": "bad", "arguments": "{not-json"}},
                    {"id": "call_good", "function": {"name": "good", "arguments": "{}"}},
                ],
            },
            {"content": "已处理这一批工具调用。", "tool_calls": []},
        ]
    )

    answer, messages = run_single_tool_turn(
        [{"role": "user", "content": "测试同批错误"}],
        model,
        {"good": lambda: called.append("good") or {"ok": True}},
        [],
    )

    assert answer == "已处理这一批工具调用。"
    assert called == ["good"]
    assert json.loads(messages[2]["content"])["error"]["type"] == "JSONDecodeError"
    assert json.loads(messages[3]["content"]) == {"ok": True}


def test_single_tool_turn_stops_after_max_tool_rounds():
    response = {
        "content": None,
        "tool_calls": [
            {"id": "call_loop", "function": {"name": "demo", "arguments": "{}"}},
        ],
    }
    model = FakeModel([response, response, response])
    called = []

    with pytest.raises(AgentLoopLimitError, match="最大轮数（2）"):
        run_single_tool_turn(
            [{"role": "user", "content": "持续调用"}],
            model,
            {"demo": lambda: called.append("demo") or {"ok": True}},
            [],
            max_tool_rounds=2,
        )

    assert called == ["demo", "demo"]
    assert len(model.calls) == 3


def test_single_tool_turn_counts_a_batch_as_one_tool_round():
    model = FakeModel(
        [
            {
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "first", "arguments": "{}"}},
                    {"id": "call_2", "function": {"name": "second", "arguments": "{}"}},
                ],
            },
            {"content": "完成。", "tool_calls": []},
        ]
    )

    answer, _ = run_single_tool_turn(
        [{"role": "user", "content": "执行一批工具"}],
        model,
        {"first": lambda: {"ok": 1}, "second": lambda: {"ok": 2}},
        [],
        max_tool_rounds=1,
    )

    assert answer == "完成。"


def test_single_tool_turn_requires_positive_max_tool_rounds():
    with pytest.raises(ValueError, match="大于或等于 1"):
        run_single_tool_turn([], FakeModel([]), {}, [], max_tool_rounds=0)


def test_single_tool_turn_emits_jsonl_events_when_logger_is_provided():
    model = FakeModel(
        [
            {
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "demo", "arguments": "{}"}},
                ],
                "usage": {"total_tokens": 12},
            },
            {"content": "完成。", "tool_calls": []},
        ]
    )
    events = []

    run_single_tool_turn(
        [{"role": "user", "content": "记录日志"}],
        model,
        {"demo": lambda: {"ok": True}},
        [],
        logger=events.append,
    )

    assert [event["event"] for event in events] == [
        "model_request",
        "model_response",
        "tool_result",
        "model_request",
        "model_response",
    ]
    assert events[0]["messages"] == [{"role": "user", "content": "记录日志"}]
    assert events[1]["usage"] == {"total_tokens": 12}
    assert events[2]["tool_call_id"] == "call_1"
    assert events[2]["result"] == '{"ok": true}'
