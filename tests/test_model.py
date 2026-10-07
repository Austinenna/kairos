from types import SimpleNamespace

from kairos.model import DeepSeekModel


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=FakeCompletions(response))


def settings():
    return SimpleNamespace(
        kairos_model="deepseek-flash",
        deepseek_api_key="test-key",
        deepseek_base_url="https://example.test",
    )


def test_complete_normalizes_direct_answer_and_forwards_request():
    client = FakeClient(
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="你好", tool_calls=[]))])
    )
    model = DeepSeekModel(settings(), client=client)
    messages = [{"role": "user", "content": "你好"}]
    tools = [{"type": "function", "function": {"name": "list_todos"}}]

    result = model.complete(messages=messages, tools=tools)

    assert result == {"content": "你好", "tool_calls": []}
    assert client.chat.completions.calls == [
        {
            "model": "deepseek-flash",
            "messages": messages,
            "tools": tools,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
    ]


def test_complete_normalizes_tool_call_and_reasoning_content():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    reasoning_content="先查待办。",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            type="function",
                            function=SimpleNamespace(name="list_todos", arguments='{"status":"pending"}'),
                        )
                    ],
                )
            )
        ]
    )
    model = DeepSeekModel(settings(), client=FakeClient(response))

    result = model.complete(messages=[], tools=[])

    assert result == {
        "content": None,
        "reasoning_content": "先查待办。",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "list_todos", "arguments": '{"status":"pending"}'},
            }
        ],
    }


def test_complete_preserves_finish_reason():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(content="部分回答", tool_calls=[]),
            )
        ]
    )
    model = DeepSeekModel(settings(), client=FakeClient(response))

    result = model.complete(messages=[], tools=[])

    assert result == {"content": "部分回答", "tool_calls": [], "finish_reason": "length"}
