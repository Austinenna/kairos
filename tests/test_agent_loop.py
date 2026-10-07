import json

from kairos.agent_loop import run_single_tool_turn


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
