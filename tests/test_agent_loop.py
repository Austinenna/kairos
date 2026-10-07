import json

import pytest

from kairos.agent_loop import AgentLoopLimitError, run_tool_loop


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, *, messages, tools):
        self.calls.append({"messages": list(messages), "tools": tools})
        return next(self.responses)


def tool_call(call_id, name, arguments="{}"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def test_tool_loop_executes_tool_and_returns_final_answer():
    model = FakeModel(
        [
            {"content": None, "tool_calls": [tool_call("call_1", "add_todo", '{"title": "学习 Agent"}')]},
            {"content": "已经帮你加好待办了。", "tool_calls": []},
        ]
    )
    called_arguments = []

    def add_todo(**arguments):
        called_arguments.append(arguments)
        return {"id": 1, "title": arguments["title"]}

    messages = [{"role": "user", "content": "把学习 Agent 加入待办"}]
    final_content, updated_messages = run_tool_loop(
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


def test_tool_loop_can_return_direct_model_answer():
    model = FakeModel([{"content": "今天没有需要处理的待办。", "tool_calls": []}])
    messages = [{"role": "user", "content": "今天有什么事？"}]

    final_content, updated_messages = run_tool_loop(messages, model, {}, [])

    assert final_content == "今天没有需要处理的待办。"
    assert updated_messages[-1] == {"role": "assistant", "content": "今天没有需要处理的待办。"}
    assert len(model.calls) == 1


def test_tool_loop_preserves_reasoning_content_in_assistant_message():
    model = FakeModel(
        [
            {
                "content": None,
                "reasoning_content": "需要先创建待办。",
                "tool_calls": [tool_call("call_1", "add_todo")],
            },
            {"content": "完成。", "tool_calls": []},
        ]
    )

    _, updated_messages = run_tool_loop(
        [{"role": "user", "content": "加一个待办"}],
        model,
        {"add_todo": lambda: {"id": 1}},
        [],
    )

    assert updated_messages[1]["reasoning_content"] == "需要先创建待办。"


def test_tool_loop_executes_multiple_tool_calls_in_one_response():
    model = FakeModel(
        [
            {
                "content": None,
                "tool_calls": [
                    tool_call("call_1", "first", '{"value": 1}'),
                    tool_call("call_2", "second", '{"value": 2}'),
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

    final_content, messages = run_tool_loop(
        [{"role": "user", "content": "执行两个操作"}],
        model,
        {"first": handler("first"), "second": handler("second")},
        [],
    )

    assert final_content == "两个操作都完成了。"
    assert called == [("first", {"value": 1}), ("second", {"value": 2})]
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "tool", "assistant"]
    assert [message["tool_call_id"] for message in messages[2:4]] == ["call_1", "call_2"]


def test_tool_loop_continues_after_multiple_tool_rounds():
    model = FakeModel(
        [
            {"content": None, "tool_calls": [tool_call("call_1", "first")]},
            {"content": None, "tool_calls": [tool_call("call_2", "second")]},
            {"content": "所有步骤完成。", "tool_calls": []},
        ]
    )
    called = []

    final_content, messages = run_tool_loop(
        [{"role": "user", "content": "分两步执行"}],
        model,
        {
            "first": lambda: called.append("first") or {"ok": 1},
            "second": lambda: called.append("second") or {"ok": 2},
        },
        [],
        max_tool_rounds=2,
    )

    assert final_content == "所有步骤完成。"
    assert called == ["first", "second"]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]


def test_tool_loop_returns_handler_error_to_model_and_continues():
    model = FakeModel(
        [
            {"content": None, "tool_calls": [tool_call("call_1", "broken")]},
            {"content": "我发现工具执行失败，请检查参数。", "tool_calls": []},
        ]
    )

    def broken():
        raise ValueError("参数不合法")

    final_content, messages = run_tool_loop(
        [{"role": "user", "content": "执行一个操作"}],
        model,
        {"broken": broken},
        [],
    )

    assert final_content == "我发现工具执行失败，请检查参数。"
    error = json.loads(messages[2]["content"])
    assert error == {"error": {"type": "ValueError", "message": "参数不合法"}}
    assert json.loads(model.calls[1]["messages"][2]["content"]) == error


@pytest.mark.parametrize(
    ("call", "expected_type"),
    [
        (tool_call("call_1", "missing", "{}"), "ValueError"),
        (tool_call("call_1", "first", "{not-json"), "JSONDecodeError"),
    ],
)
def test_tool_loop_returns_argument_or_unknown_tool_errors(call, expected_type):
    model = FakeModel(
        [
            {"content": None, "tool_calls": [call]},
            {"content": "已收到错误。", "tool_calls": []},
        ]
    )

    _, messages = run_tool_loop(
        [{"role": "user", "content": "执行操作"}],
        model,
        {"first": lambda: {"ok": True}},
        [],
    )

    error = json.loads(messages[2]["content"])
    assert error["error"]["type"] == expected_type


def test_tool_loop_rejects_infinite_tool_rounds():
    response = {"content": None, "tool_calls": [tool_call("call_1", "first")]}
    model = FakeModel([response, response, response])

    with pytest.raises(AgentLoopLimitError, match="最大轮数（2）"):
        run_tool_loop(
            [{"role": "user", "content": "一直执行"}],
            model,
            {"first": lambda: {"ok": True}},
            [],
            max_tool_rounds=2,
        )

    assert len(model.calls) == 3


def test_tool_loop_marks_truncated_final_response():
    model = FakeModel([{"content": "部分回答", "tool_calls": [], "finish_reason": "length"}])

    final_content, _ = run_tool_loop([{"role": "user", "content": "回答"}], model, {}, [])

    assert final_content.endswith("（模型输出达到长度上限，以上内容可能不完整。）")


def test_tool_loop_requires_positive_max_tool_rounds():
    with pytest.raises(ValueError, match="大于或等于 1"):
        run_tool_loop([], FakeModel([]), {}, [], max_tool_rounds=0)
