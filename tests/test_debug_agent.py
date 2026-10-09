import json
from types import SimpleNamespace

from kairos.agent_app import run_cli, run_prompt


class FakeModel:
    def __init__(self):
        self.calls = []
        self.responses = iter(
            [
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "add_todo", "arguments": '{"title":"真实链路测试"}'},
                        }
                    ],
                },
                {"content": "已加入待办。", "tool_calls": []},
            ]
        )

    def complete(self, *, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        return next(self.responses)


def test_run_prompt_wires_model_to_sqlite_and_tool_schemas(tmp_path):
    model = FakeModel()

    answer, messages = run_prompt("加一个待办", model=model, db_path=tmp_path / "kairos.sqlite3")

    assert answer == "已加入待办。"
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert model.calls[0]["tools"][0]["function"]["name"] == "add_todo"


def test_run_prompt_can_write_jsonl_observability_log(tmp_path):
    model = FakeModel()
    log_path = tmp_path / "logs" / "agent.jsonl"

    run_prompt("加一个待办", model=model, db_path=tmp_path / "kairos.sqlite3", log_path=log_path)

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == [
        "model_request",
        "model_response",
        "tool_result",
        "model_request",
        "model_response",
    ]


class FakeCompletions:
    def __init__(self, responses):
        self.responses = iter(responses)

    def create(self, **kwargs):
        return next(self.responses)


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


class FakeAnthropicMessages:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


class FakeAnthropicClient:
    def __init__(self, responses):
        self.messages = FakeAnthropicMessages(responses)


def test_run_cli_connects_real_adapter_to_tool_loop(tmp_path):
    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                type="function",
                                function=SimpleNamespace(name="add_todo", arguments='{"title":"适配器测试"}'),
                            )
                        ],
                    )
                )
            ]
        ),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="已完成。", tool_calls=[]))]),
    ]
    settings = SimpleNamespace(
        kairos_model="deepseek-flash",
        deepseek_api_key="test-key",
        deepseek_base_url="https://example.test",
    )

    answer = run_cli(
        "加一个待办",
        settings=settings,
        client=FakeClient(responses),
        db_path=tmp_path / "kairos.sqlite3",
    )

    assert answer == "已完成。"


def test_run_cli_can_select_anthropic_messages_adapter(tmp_path):
    client = FakeAnthropicClient(
        [
            {
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "add_todo", "input": {"title": "Anthropic 链路"}}
                ]
            },
            {"content": [{"type": "text", "text": "已加入待办。"}]},
        ]
    )
    settings = SimpleNamespace(
        kairos_provider="anthropic",
        kairos_model="claude-test",
        anthropic_api_key="test-key",
        anthropic_base_url="https://gateway.example",
        anthropic_max_tokens=128,
    )

    answer = run_cli(
        "加一个待办",
        settings=settings,
        client=client,
        db_path=tmp_path / "kairos.sqlite3",
    )

    assert answer == "已加入待办。"
    assert client.messages.calls[0]["model"] == "claude-test"
    assert client.messages.calls[0]["max_tokens"] == 128
