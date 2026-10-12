from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletionChunk

from kairos.providers import OpenAICompatibleModel


class FakeStream:
    """可观测读取时序，并能在尾部模拟传输失败的同步 SDK 流。"""

    def __init__(self, chunks):
        self.chunks = chunks
        self.read_count = 0
        self.drained = False
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            self.read_count += 1
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk
        self.drained = True

    def close(self):
        self.closed = True


def make_model(chunks, **options):
    stream = FakeStream(chunks)
    requests = []

    def create(**request):
        requests.append(request)
        return stream

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return OpenAICompatibleModel(model="test-model", client=client, **options), stream, requests


def chunk(delta=None, finish_reason=None):
    return {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}]}


def tool_delta(index=0, *, call_id="call_1", name="add_todo", arguments='{"title":"学习"}'):
    return {
        "index": index,
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def test_stream_calls_back_before_drain_and_reads_usage_after_finish():
    sdk_chunk = ChatCompletionChunk(
        id="response_1",
        created=0,
        model="test-model",
        object="chat.completion.chunk",
        choices=[{"index": 0, "delta": {"content": "你", "reasoning_content": "思考"}}],
    )
    model, stream, requests = make_model(
        [
            chunk({"role": "assistant", "content": ""}),
            sdk_chunk,
            chunk({"content": "好", "reasoning_content": "完成"}),
            chunk(finish_reason="stop"),
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 6,
                    "total_tokens": 16,
                    "prompt_tokens_details": {"cached_tokens": 3},
                    "completion_tokens_details": {"reasoning_tokens": 4},
                },
            },
        ],
        extra_body={"thinking": {"type": "enabled"}},
    )
    visible = []

    def on_text(text):
        assert not stream.drained
        assert not stream.closed
        visible.append((text, stream.read_count))

    messages = [{"role": "user", "content": "问好"}]
    result = model.stream(messages=messages, tools=[], on_text=on_text)

    assert visible == [("你", 2), ("好", 3)]
    assert result == {
        "content": "你好",
        "reasoning_content": "思考完成",
        "tool_calls": [],
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 6,
            "total_tokens": 16,
            "cached_input_tokens": 3,
            "reasoning_tokens": 4,
        },
    }
    assert requests == [
        {
            "model": "test-model",
            "messages": messages,
            "tools": [],
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": {"thinking": {"type": "enabled"}},
        }
    ]
    assert stream.drained and stream.closed


def test_stream_assembles_interleaved_tools_in_index_order_after_full_response():
    model, stream, _ = make_model(
        [
            chunk({"tool_calls": [tool_delta(1, call_id="call_", name="sec", arguments='{"value":')]}),
            chunk({"content": "准备执行。", "tool_calls": [tool_delta(0, name="first", arguments='{"value":')]}),
            chunk({"tool_calls": [tool_delta(1, call_id="2", name="ond", arguments="2}")]}),
            chunk({"tool_calls": [tool_delta(0, call_id=None, name=None, arguments="1}")]}),
            chunk(finish_reason="tool_calls"),
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
        ]
    )
    seen = []
    result = model.stream(messages=[], tools=[], on_text=seen.append)

    assert result["tool_calls"] == [
        {"id": "call_1", "type": "function", "function": {"name": "first", "arguments": '{"value":1}'}},
        {"id": "call_2", "type": "function", "function": {"name": "second", "arguments": '{"value":2}'}},
    ]
    assert result["usage"]["total_tokens"] == 5
    assert seen == ["准备执行。"]
    assert stream.drained and stream.closed


@pytest.mark.parametrize("cached", [0, 7])
def test_stream_preserves_deepseek_cache_usage(cached):
    model, _, _ = make_model(
        [
            chunk(finish_reason="stop"),
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                    "prompt_cache_hit_tokens": cached,
                    "prompt_tokens_details": {"cached_tokens": 8},
                },
            },
        ]
    )
    assert model.stream(messages=[], tools=[], on_text=lambda _: None)["usage"]["cached_input_tokens"] == cached


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_stream_rejects_truncated_tool_even_when_arguments_are_valid_json(finish_reason):
    model, stream, _ = make_model(
        [
            chunk({"tool_calls": [tool_delta(arguments="{}")]}),
            chunk(finish_reason=finish_reason),
        ]
    )
    with pytest.raises(ValueError, match="工具调用未完整结束"):
        model.stream(messages=[], tools=[], on_text=lambda _: None)
    assert stream.drained and stream.closed


def test_stream_returns_length_limited_text_for_renderer():
    model, stream, _ = make_model([chunk({"content": "未完"}), chunk(finish_reason="length")])
    assert model.stream(messages=[], tools=[], on_text=lambda _: None) == {
        "content": "未完",
        "tool_calls": [],
        "finish_reason": "length",
    }
    assert stream.closed


@pytest.mark.parametrize("tools", [[], [tool_delta(arguments="{}")]])
def test_stream_rejects_missing_finish_reason(tools):
    model, stream, _ = make_model([chunk({"content": "部分正文", "tool_calls": tools})])
    with pytest.raises(ValueError, match="缺少 finish_reason"):
        model.stream(messages=[], tools=[], on_text=lambda _: None)
    assert stream.closed


@pytest.mark.parametrize(
    "calls, message",
    [
        ([tool_delta(call_id=None)], "缺少 id 或 name"),
        ([tool_delta(name=" ")], "缺少 id 或 name"),
        ([tool_delta(), tool_delta(1)], "重复"),
        ([tool_delta(arguments='{"title":')], "完整的 JSON 对象"),
        ([tool_delta(arguments="")], "完整的 JSON 对象"),
        ([tool_delta(arguments="[]")], "必须是 JSON 对象"),
        ([tool_delta(arguments="null")], "必须是 JSON 对象"),
        ([tool_delta(arguments='{"value":NaN}')], "完整的 JSON 对象"),
        ([tool_delta(arguments="{} {}")], "完整的 JSON 对象"),
        ([tool_delta(index=-1)], "index"),
    ],
)
def test_stream_rejects_unusable_tool_calls(calls, message):
    model, stream, _ = make_model([chunk({"tool_calls": calls}), chunk(finish_reason="tool_calls")])
    with pytest.raises(ValueError, match=message):
        model.stream(messages=[], tools=[], on_text=lambda _: None)
    assert stream.closed


@pytest.mark.parametrize("error", [RuntimeError("断流"), KeyboardInterrupt()])
@pytest.mark.parametrize("after_finish", [False, True])
def test_stream_propagates_read_failure_and_closes_even_after_finish(error, after_finish):
    chunks = [chunk({"tool_calls": [tool_delta()]})]
    if after_finish:
        chunks.append(chunk(finish_reason="tool_calls"))
    model, stream, _ = make_model([*chunks, error])
    with pytest.raises(type(error)) as raised:
        model.stream(messages=[], tools=[], on_text=lambda _: None)
    assert raised.value is error
    assert stream.closed


@pytest.mark.parametrize("error", [RuntimeError("显示失败"), KeyboardInterrupt()])
def test_stream_closes_when_callback_fails(error):
    model, stream, _ = make_model([chunk({"content": "正文"}), chunk(finish_reason="stop")])

    def fail(_):
        raise error

    with pytest.raises(type(error)) as raised:
        model.stream(messages=[], tools=[], on_text=fail)
    assert raised.value is error
    assert stream.closed and not stream.drained


def test_stream_rejects_tool_calls_finish_without_calls():
    model, stream, _ = make_model([chunk(finish_reason="tool_calls")])
    with pytest.raises(ValueError, match="没有工具调用"):
        model.stream(messages=[], tools=[], on_text=lambda _: None)
    assert stream.closed


def test_stream_rejects_content_after_finish():
    model, stream, _ = make_model([chunk(finish_reason="stop"), chunk({"tool_calls": [tool_delta()]})])
    with pytest.raises(ValueError, match="finish_reason 之后"):
        model.stream(messages=[], tools=[], on_text=lambda _: None)
    assert stream.closed
