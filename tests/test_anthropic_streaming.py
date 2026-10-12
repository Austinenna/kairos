import copy
import json
from types import SimpleNamespace

import pytest
from anthropic.types import RawMessageDeltaEvent, RawMessageStartEvent

from kairos.providers.anthropic_messages import AnthropicMessagesModel, anthropic_messages


class FakeStream:
    def __init__(self, events):
        self.events = iter(events)
        self.closed = False

    def __iter__(self):
        for event in self.events:
            if isinstance(event, BaseException):
                raise event
            yield event

    def close(self):
        self.closed = True


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        return self.response


def model_with(response):
    messages = FakeMessages(response)
    model = AnthropicMessagesModel("test-model", client=SimpleNamespace(messages=messages), max_tokens=321)
    return model, messages


def start(usage=None):
    return {"type": "message_start", "message": {"content": [], "usage": usage}}


def block_start(index, **block):
    return {"type": "content_block_start", "index": index, "content_block": block}


def delta(index, kind, **values):
    return {"type": "content_block_delta", "index": index, "delta": {"type": kind, **values}}


def block_stop(index):
    return {"type": "content_block_stop", "index": index}


def end(reason="end_turn", usage=None):
    return [
        {"type": "message_delta", "delta": {"stop_reason": reason}, "usage": usage},
        {"type": "message_stop"},
    ]


def text_block(index, text):
    return [block_start(index, type="text", text=""), delta(index, "text_delta", text=text), block_stop(index)]


def tool_block(index, call_id="tool_1", name="add_todo", arguments='{"title":"学习"}'):
    return [
        block_start(index, type="tool_use", id=call_id, name=name, input={}),
        delta(index, "input_json_delta", partial_json=arguments[:9]),
        delta(index, "input_json_delta", partial_json=arguments[9:]),
        block_stop(index),
    ]


def stream_events(events, on_text=None):
    response = FakeStream(events)
    model, _ = model_with(response)
    result = model.stream(messages=[], tools=[], on_text=on_text or (lambda _: None))
    assert response.closed
    return result


def test_text_is_emitted_before_stream_finishes_and_request_parameters_are_preserved():
    seen = []

    def events():
        yield start()
        yield block_start(0, type="text", text="开头")
        assert seen == ["开头"]
        yield delta(0, "text_delta", text="正文")
        assert seen == ["开头", "正文"]
        yield block_stop(0)
        yield from end()

    response = FakeStream(events())
    model, messages = model_with(response)
    tools = [{"type": "function", "function": {"name": "demo", "parameters": {"type": "object"}}}]
    result = model.stream(
        messages=[{"role": "system", "content": "提示词"}, {"role": "user", "content": "问题"}],
        tools=tools,
        on_text=seen.append,
    )

    assert result["content"] == "开头正文"
    assert result["tool_calls"] == []
    assert response.closed
    assert messages.requests == [
        {
            "model": "test-model",
            "max_tokens": 321,
            "messages": [{"role": "user", "content": "问题"}],
            "tools": [{"name": "demo", "description": "", "input_schema": {"type": "object"}}],
            "system": "提示词",
            "stream": True,
        }
    ]


def test_interleaved_text_and_multiple_tools_preserve_order_and_complete_arguments():
    seen = []
    events = [
        start(),
        *text_block(0, "先处理第一项"),
        *tool_block(1),
        *text_block(2, "再处理第二项"),
        *tool_block(3, "tool_2", "list_todos", "{}"),
        *end("tool_use"),
    ]

    result = stream_events(events, seen.append)

    assert "".join(seen) == result["content"] == "先处理第一项\n再处理第二项"
    assert [call["id"] for call in result["tool_calls"]] == ["tool_1", "tool_2"]
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"title": "学习"}
    assert json.loads(result["tool_calls"][1]["function"]["arguments"]) == {}
    assert [block["type"] for block in result["provider_content"]] == ["text", "tool_use", "text", "tool_use"]


def test_usage_combines_sdk_start_and_cumulative_deltas_without_erasing_input():
    message = RawMessageStartEvent.model_validate(
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "test-model",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 1},
            },
        }
    )
    # SDK 为省略的 input_tokens 提供 None，不能覆盖 message_start 中的 12。
    first = RawMessageDeltaEvent.model_validate(
        {
            "type": "message_delta",
            "delta": {"stop_reason": None, "stop_sequence": None},
            "usage": {"output_tokens": 3},
        }
    )
    last = RawMessageDeltaEvent.model_validate(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 7},
        }
    )

    result = stream_events([message, *text_block(0, "完成"), first, last, {"type": "message_stop"}])

    assert result["usage"] == {
        "prompt_tokens": 12,
        "cached_input_tokens": None,
        "completion_tokens": 7,
        "total_tokens": 19,
    }


def test_cached_usage_matches_complete_and_preserves_optional_fields_across_cumulative_deltas():
    initial_usage = {
        "input_tokens": 12,
        "output_tokens": 1,
        "cache_read_input_tokens": 20,
        "cache_creation_input_tokens": 3,
    }
    deltas = [
        RawMessageDeltaEvent.model_validate(
            {
                "type": "message_delta",
                "delta": {"stop_reason": reason, "stop_sequence": None},
                "usage": usage,
            }
        )
        for reason, usage in [
            (None, {"output_tokens": 3, "cache_read_input_tokens": 22, "cache_creation_input_tokens": 4}),
            (None, {"input_tokens": 15, "output_tokens": 6}),
            ("end_turn", {"output_tokens": 7}),
        ]
    ]
    streamed = stream_events([start(initial_usage), *text_block(0, "完成"), *deltas, {"type": "message_stop"}])
    model, _ = model_with(
        {
            "content": [{"type": "text", "text": "完成"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 15,
                "output_tokens": 7,
                "cache_read_input_tokens": 22,
                "cache_creation_input_tokens": 4,
            },
        }
    )
    completed = model.complete(messages=[], tools=[])

    assert (
        streamed["usage"]
        == completed["usage"]
        == {
            "prompt_tokens": 41,
            "cached_input_tokens": 22,
            "completion_tokens": 7,
            "total_tokens": 48,
        }
    )


def test_signed_thinking_and_redacted_blocks_are_replayed_in_original_order():
    seen = []
    result = stream_events(
        [
            start(),
            block_start(0, type="thinking", thinking="", signature=""),
            delta(0, "thinking_delta", thinking="先思考"),
            delta(0, "thinking_delta", thinking="再决定"),
            delta(0, "signature_delta", signature="opaque-"),
            delta(0, "signature_delta", signature="signature"),
            block_stop(0),
            block_start(1, type="redacted_thinking", data="opaque-data"),
            block_stop(1),
            *text_block(2, "调用工具"),
            *tool_block(3),
            *end("tool_use"),
        ],
        seen.append,
    )

    expected = [
        {"type": "thinking", "thinking": "先思考再决定", "signature": "opaque-signature"},
        {"type": "redacted_thinking", "data": "opaque-data"},
        {"type": "text", "text": "调用工具"},
        {"type": "tool_use", "id": "tool_1", "name": "add_todo", "input": {"title": "学习"}},
    ]
    assert seen == ["调用工具"]
    assert result["reasoning_content"] == "先思考再决定"
    assert result["provider_content"] == expected
    history = [{"role": "assistant", **result}, {"role": "tool", "tool_call_id": "tool_1", "content": "成功"}]
    _, replay = anthropic_messages(history)
    assert replay[0] == {"role": "assistant", "content": expected}
    assert replay[1]["content"] == [{"type": "tool_result", "tool_use_id": "tool_1", "content": "成功"}]
    replay[0]["content"][0]["signature"] = "changed"
    assert history[0]["provider_content"][0]["signature"] == "opaque-signature"


def test_complete_preserves_sdk_content_and_accepts_provider_content_history():
    from anthropic.types import Message

    blocks = [
        {"type": "thinking", "thinking": "思考", "signature": "signed"},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "text", "text": "回答"},
    ]
    response = Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "test-model",
            "content": blocks,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }
    )
    model, client_messages = model_with(response)
    result = model.complete(messages=[{"role": "assistant", "content": "摘要", "provider_content": blocks}], tools=[])

    assert result["provider_content"] == blocks
    assert client_messages.requests[0]["messages"] == [{"role": "assistant", "content": blocks}]
    assert "stream" not in client_messages.requests[0]
    assert result["content"] == "回答"


@pytest.mark.parametrize(
    "events",
    [
        [start(), *tool_block(0), end("tool_use")[0]],
        [start(), *tool_block(0), {"type": "message_stop"}],
        [start(), *tool_block(0)[:-1], *end("tool_use")],
        [start(), *tool_block(0), block_start(1, type="text", text="未结束"), *end("tool_use")],
        [*tool_block(0), *end("tool_use")],
    ],
)
def test_missing_terminal_events_reject_tools_and_close_stream(events):
    response = FakeStream(events)
    model, _ = model_with(response)
    with pytest.raises(ValueError):
        model.stream(messages=[], tools=[], on_text=lambda _: None)
    assert response.closed


@pytest.mark.parametrize("arguments", ['{"title":', "[]", '"string"', "null", '{"x": NaN}', "{}{}"])
def test_invalid_or_incomplete_tool_json_is_rejected(arguments):
    response = FakeStream([start(), *tool_block(0, arguments=arguments), *end("tool_use")])
    model, _ = model_with(response)
    with pytest.raises(ValueError):
        model.stream(messages=[], tools=[], on_text=lambda _: None)
    assert response.closed


@pytest.mark.parametrize(("call_id", "name"), [(None, "demo"), ("", "demo"), (7, "demo"), ("a", None), ("a", " ")])
def test_tools_require_nonempty_string_id_and_name(call_id, name):
    with pytest.raises(ValueError, match="id、name"):
        stream_events([start(), *tool_block(0, call_id, name), *end("tool_use")])


def test_duplicate_tool_ids_are_rejected_before_returning_any_calls():
    with pytest.raises(ValueError, match="id 重复"):
        stream_events([start(), *tool_block(0), *tool_block(1), *end("tool_use")])


def test_no_argument_tool_can_finish_without_input_deltas():
    result = stream_events(
        [
            start(),
            block_start(0, type="tool_use", id="tool_1", name="list_todos", input={}),
            block_stop(0),
            *end("tool_use"),
        ]
    )
    assert result["tool_calls"][0]["function"]["arguments"] == "{}"


def test_tool_without_initial_input_or_argument_deltas_is_rejected():
    with pytest.raises(ValueError, match="对象参数"):
        stream_events(
            [
                start(),
                block_start(0, type="tool_use", id="tool_1", name="list_todos"),
                block_stop(0),
                *end("tool_use"),
            ]
        )


def test_legacy_history_without_provider_content_still_converts_tools():
    system, messages = anthropic_messages(
        [
            {"role": "system", "content": "提示"},
            {
                "role": "assistant",
                "content": "调用",
                "tool_calls": [{"id": "tool_1", "function": {"name": "demo", "arguments": '{"x":1}'}}],
            },
        ]
    )
    assert system == "提示"
    assert messages == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "调用"},
                {"type": "tool_use", "id": "tool_1", "name": "demo", "input": {"x": 1}},
            ],
        }
    ]


@pytest.mark.parametrize("reason", ["end_turn", "max_tokens", "stop_sequence"])
def test_tools_require_tool_use_stop_reason(reason):
    with pytest.raises(ValueError, match="结束原因不一致"):
        stream_events([start(), *tool_block(0), *end(reason)])


def test_tool_use_stop_reason_requires_tool_content():
    with pytest.raises(ValueError, match="结束原因不一致"):
        stream_events([start(), *text_block(0, "正文"), *end("tool_use")])


def test_text_without_tools_may_finish_at_max_tokens():
    result = stream_events([start(), *text_block(0, "未写完"), *end("max_tokens")])
    assert result["content"] == "未写完"
    assert result["finish_reason"] == "max_tokens"
    assert result["tool_calls"] == []


@pytest.mark.parametrize("error", [RuntimeError("连接中断"), KeyboardInterrupt()])
def test_stream_closes_on_transport_error_or_keyboard_interrupt(error):
    response = FakeStream([start(), *text_block(0, "已输出"), error])
    model, _ = model_with(response)
    with pytest.raises(type(error)):
        model.stream(messages=[], tools=[], on_text=lambda _: None)
    assert response.closed


def test_stream_closes_if_text_callback_raises():
    def fail(_):
        raise RuntimeError("显示失败")

    response = FakeStream([start(), *text_block(0, "正文"), *end()])
    model, _ = model_with(response)
    with pytest.raises(RuntimeError, match="显示失败"):
        model.stream(messages=[], tools=[], on_text=fail)
    assert response.closed


@pytest.mark.parametrize(
    "events",
    [
        [start(), delta(0, "text_delta", text="未开始"), *end()],
        [start(), *text_block(0, "结束"), delta(0, "text_delta", text="太晚"), *end()],
        [start(), block_start(0, type="text", text=""), delta(0, "input_json_delta", partial_json="{}"), *end()],
        [start(), *text_block(0, "正文"), block_start(0, type="text", text="重复"), *end()],
    ],
)
def test_invalid_block_lifecycle_or_delta_type_is_rejected(events):
    with pytest.raises(ValueError):
        stream_events(events)
