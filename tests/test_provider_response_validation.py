"""验证两种协议在流式与非流式下共享工具响应的执行边界。"""

import json
from types import SimpleNamespace

import httpx2
import pytest
from anthropic import Anthropic
from openai import OpenAI

from kairos.providers import AnthropicMessagesModel, OpenAICompatibleModel
from kairos.runtime import create_runtime
from kairos_todo import list_todos


class FakeStream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def __iter__(self):
        yield from self.events

    def close(self):
        self.closed = True


def tool_call(call_id="add_1", **overrides):
    call = {
        "id": call_id,
        "type": "function",
        "function": {"name": "add_todo", "arguments": '{"title":"不应提前写入"}'},
    }
    call.update(overrides)
    return call


def response_for(provider, reason, calls):
    if provider == "openai":
        return {
            "id": "response_test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": reason,
                    "message": {"role": "assistant", "content": "正文", "tool_calls": calls},
                }
            ],
        }
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "test-model",
        "content": [
            {"type": "text", "text": "正文"},
            *[
                {
                    "type": "tool_use",
                    "id": call["id"],
                    "name": call["function"]["name"],
                    "input": json.loads(call["function"]["arguments"]),
                }
                for call in calls
            ],
        ],
        "stop_reason": reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def call_model(provider, stream, reason, calls):
    response = response_for(provider, reason, calls)
    if stream:
        if provider == "openai":
            message = response["choices"][0]["message"]
            response = FakeStream(
                [
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": message["content"],
                                    "tool_calls": [{"index": index, **call} for index, call in enumerate(calls)],
                                },
                                "finish_reason": None,
                            }
                        ]
                    },
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]},
                ]
            )
        else:
            events = [{"type": "message_start", "message": {"content": []}}]
            for index, block in enumerate(response["content"]):
                events.extend(
                    [
                        {"type": "content_block_start", "index": index, "content_block": block},
                        {"type": "content_block_stop", "index": index},
                    ]
                )
            events.extend([{"type": "message_delta", "delta": {"stop_reason": reason}}, {"type": "message_stop"}])
            response = FakeStream(events)
    endpoint = SimpleNamespace(create=lambda **_: response)
    model = (
        OpenAICompatibleModel(model="test-model", client=SimpleNamespace(chat=SimpleNamespace(completions=endpoint)))
        if provider == "openai"
        else AnthropicMessagesModel("test-model", client=SimpleNamespace(messages=endpoint))
    )
    try:
        if stream:
            return model.stream(messages=[], tools=[], on_text=lambda _: None)
        return model.complete(messages=[], tools=[])
    finally:
        if stream:
            assert response.closed


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("provider,reason", [("openai", "tool_calls"), ("openai", "stop"), ("anthropic", "tool_use")])
def test_complete_tool_response_accepts_same_calls_in_both_modes(provider, stream, reason):
    result = call_model(provider, stream, reason, [tool_call()])
    assert result["tool_calls"][0]["id"] == "add_1"
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"title": "不应提前写入"}
    assert result["finish_reason"] == reason


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "provider,reason",
    [
        ("openai", "length"),
        ("openai", "content_filter"),
        ("openai", None),
        ("openai", "unknown"),
        ("anthropic", "max_tokens"),
        ("anthropic", "end_turn"),
        ("anthropic", "stop_sequence"),
        ("anthropic", None),
    ],
)
def test_complete_json_does_not_make_incomplete_tool_response_executable(provider, stream, reason):
    with pytest.raises(ValueError):
        call_model(provider, stream, reason, [tool_call()])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("provider,reason", [("openai", "tool_calls"), ("anthropic", "tool_use")])
def test_tool_finish_reason_requires_actual_tools(provider, stream, reason):
    with pytest.raises(ValueError):
        call_model(provider, stream, reason, [])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "provider,reason", [("openai", "length"), ("openai", "content_filter"), ("anthropic", "max_tokens")]
)
def test_truncated_text_without_tools_remains_available_for_display(provider, stream, reason):
    result = call_model(provider, stream, reason, [])
    assert result["content"] == "正文"
    assert result["tool_calls"] == []
    assert result["finish_reason"] == reason


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_text_response_also_requires_end_reason(provider, stream):
    with pytest.raises(ValueError):
        call_model(provider, stream, None, [])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("arguments", ['{"title":', "", "[]", "null", '{"x":NaN}', '{"x":Infinity}', "{}{}"])
def test_openai_validates_every_tool_json_before_returning_batch(stream, arguments):
    calls = [tool_call(), tool_call("add_2", function={"name": "add_todo", "arguments": arguments})]
    with pytest.raises(ValueError):
        call_model("openai", stream, "tool_calls", calls)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "calls",
    [
        [tool_call(" ")],
        [tool_call(1)],
        [tool_call(function={"name": " ", "arguments": "{}"})],
        [tool_call(type="custom")],
        [tool_call(), tool_call()],
    ],
)
def test_openai_tool_structure_is_checked_in_both_modes(stream, calls):
    with pytest.raises(ValueError):
        call_model("openai", stream, "tool_calls", calls)


def test_openai_complete_rejects_legacy_function_call():
    response = response_for("openai", "stop", [])
    response["choices"][0]["message"]["function_call"] = {"name": "add_todo", "arguments": "{}"}
    endpoint = SimpleNamespace(create=lambda **_: response)
    model = OpenAICompatibleModel(
        model="test-model", client=SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
    )
    with pytest.raises(ValueError, match="旧式 function_call"):
        model.complete(messages=[], tools=[])


@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_rejects_non_json_constants_even_in_complete_input_object(stream):
    with pytest.raises(ValueError):
        call_model("anthropic", stream, "tool_use", [tool_call(function={"name": "demo", "arguments": '{"x":NaN}'})])


@pytest.mark.parametrize(
    "provider,reason,broken_last",
    [
        ("openai", "length", False),
        ("openai", "content_filter", False),
        ("openai", None, False),
        ("openai", "tool_calls", True),
        ("anthropic", "max_tokens", False),
        ("anthropic", None, False),
    ],
)
def test_sdk_nonstream_rejects_entire_batch_before_database_and_history_changes(
    tmp_path, provider, reason, broken_last
):
    calls = [tool_call(), tool_call("add_2")]
    if broken_last:
        calls[-1]["function"]["arguments"] = '{"title":'
    response = response_for(provider, reason, calls)
    transport = httpx2.MockTransport(lambda _: httpx2.Response(200, json=response))
    client_class = OpenAI if provider == "openai" else Anthropic
    with client_class(api_key="offline-test", max_retries=0, http_client=httpx2.Client(transport=transport)) as client:
        model = (
            OpenAICompatibleModel(model="test-model", client=client)
            if provider == "openai"
            else AnthropicMessagesModel("test-model", client=client)
        )
        with create_runtime(model=model, db_path=tmp_path / "test.sqlite3") as runtime:
            with pytest.raises(ValueError):
                runtime.respond("添加两项待办")
            assert list_todos(runtime.connection) == []
            assert len(runtime.history()) == 1
            details = runtime.session_store.inspect(runtime.session_id)
            assert details["turns"][0]["status"] == "failed"
            assert details["steps"][0]["status"] == "failed"
            assert details["tool_calls"] == []
            assert not any(message["role"] in {"assistant", "tool"} for message in details["messages"])
