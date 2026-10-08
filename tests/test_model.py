import json
from types import SimpleNamespace

from kairos.model import ClaudeModel, DeepSeekModel


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


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeAnthropicClient:
    def __init__(self, response):
        self.messages = FakeMessages(response)


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


def test_complete_normalizes_usage_and_finish_reason():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="完成", tool_calls=[]),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14),
    )
    model = DeepSeekModel(settings(), client=FakeClient(response))

    result = model.complete(messages=[], tools=[])

    assert result == {
        "content": "完成",
        "tool_calls": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        "finish_reason": "stop",
    }


def test_claude_complete_converts_request_and_normalizes_text_response():
    client = FakeAnthropicClient(
        SimpleNamespace(
            content=[SimpleNamespace(type="text", text="已查到。")],
            usage=SimpleNamespace(input_tokens=8, output_tokens=3),
            stop_reason="end_turn",
        )
    )
    model = ClaudeModel("claude-test", client=client, max_tokens=128)
    messages = [
        {"role": "system", "content": "你是 Kairos。"},
        {"role": "user", "content": "查一下待办"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_todos",
                "description": "查询待办",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    result = model.complete(messages=messages, tools=tools)

    assert result == {
        "content": "已查到。",
        "tool_calls": [],
        "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
        "finish_reason": "end_turn",
    }
    assert client.messages.calls == [
        {
            "model": "claude-test",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "查一下待办"}],
            "tools": [
                {
                    "name": "list_todos",
                    "description": "查询待办",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "system": "你是 Kairos。",
        }
    ]


def test_claude_complete_converts_tool_history_and_normalizes_tool_use():
    client = FakeAnthropicClient(
        {
            "content": [
                {"type": "text", "text": "我先查询。"},
                {"type": "tool_use", "id": "toolu_1", "name": "list_todos", "input": {"status": "pending"}},
                {"type": "tool_use", "id": "toolu_2", "name": "list_todos", "input": {}},
            ]
        }
    )
    model = ClaudeModel("claude-test", client=client)
    messages = [
        {"role": "user", "content": "查待办"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_old",
                    "type": "function",
                    "function": {"name": "list_todos", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_old", "content": '{"id": 1}'},
        {"role": "tool", "tool_call_id": "toolu_old_2", "content": '{"id": 2}'},
    ]

    result = model.complete(messages=messages, tools=[])

    assert [call["function"]["name"] for call in result["tool_calls"]] == ["list_todos", "list_todos"]
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"status": "pending"}
    assert client.messages.calls[0]["messages"] == [
        {"role": "user", "content": "查待办"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_old", "name": "list_todos", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_old", "content": '{"id": 1}'},
                {"type": "tool_result", "tool_use_id": "toolu_old_2", "content": '{"id": 2}'},
            ],
        },
    ]
