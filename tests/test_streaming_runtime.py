"""用真实 SDK 的离线 SSE 解码验证从片段到工具执行的完整边界。"""

import json
from types import SimpleNamespace

import httpx2
import pytest
from openai import OpenAI

from kairos.providers import OpenAICompatibleModel
from kairos.runtime import create_runtime
from kairos.todos import list_todos


class SSEBody(httpx2.SyncByteStream):
    def __init__(self, frames, *, anthropic=False):
        self.frames = frames
        self.anthropic = anthropic
        self.drained = False
        self.closed = False

    def __iter__(self):
        for frame in self.frames:
            payload = (
                frame
                if self.anthropic
                else {
                    "id": "response_test",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "test-model",
                    **frame,
                }
            )
            prefix = f"event: {frame['type']}\n" if self.anthropic else ""
            encoded = f"{prefix}data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
            # 故意把 UTF-8 字符和 SSE 行拆开，经过 SDK 解码而非直接注入响应对象。
            for index in range(0, len(encoded), 7):
                yield encoded[index : index + 7]
        self.drained = True
        if not self.anthropic:
            yield b"data: [DONE]\n\n"

    def close(self):
        self.closed = True


def delta(value=None, reason=None):
    return {"choices": [{"index": 0, "delta": value or {}, "finish_reason": reason}]}


def test_sdk_stream_displays_two_texts_then_runs_three_tools_and_continues(tmp_path):
    frames = [delta({"content": "我先添加。", "reasoning_content": "准备工具"})]
    for index in range(3):
        frames.append(
            delta(
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {"name": "add_todo", "arguments": '{"title":"'},
                        }
                    ]
                }
            )
        )
        if index == 0:
            frames.append(delta({"content": "共有三项。"}))
        for fragment in ["事项", str(index + 1), '"}']:
            frames.append(delta({"tool_calls": [{"index": index, "function": {"arguments": fragment}}]}))
    frames += [
        delta(reason="tool_calls"),
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 2},
            },
        },
    ]
    bodies = [
        SSEBody(frames),
        SSEBody(
            [
                delta({"content": "三项"}),
                delta({"content": "均已添加。"}),
                delta(reason="stop"),
                {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27}},
            ]
        ),
    ]
    requests, visible, executed = [], [], []

    def handle(request):
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["stream_options"]["include_usage"] is True
        if requests:
            assert executed == ["事项1", "事项2", "事项3"]
            tools = [message for message in body["messages"] if message["role"] == "tool"]
            assert [message["tool_call_id"] for message in tools] == ["call_0", "call_1", "call_2"]
            assert [json.loads(message["content"])["id"] for message in tools] == [1, 2, 3]
            assert body["messages"][2]["reasoning_content"] == "准备工具"
        requests.append(body)
        return httpx2.Response(200, headers={"Content-Type": "text/event-stream"}, stream=bodies[len(requests) - 1])

    with OpenAI(
        api_key="test-key", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(handle))
    ) as client:
        with create_runtime(
            model=OpenAICompatibleModel(model="test-model", client=client), db_path=tmp_path / "todos.sqlite3"
        ) as runtime:
            original_add = runtime.tool_registry.handlers["add_todo"]

            def add(title):
                assert bodies[0].drained and bodies[0].closed
                assert visible == ["我先添加。", "共有三项。"]
                executed.append(title)
                return original_add(title=title)

            runtime.tool_registry.handlers["add_todo"] = add

            def on_event(event):
                if event["event"] == "text_delta":
                    assert not bodies[len(requests) - 1].drained
                    if len(requests) == 1:
                        assert executed == []
                    visible.append(event["text"])

            result = runtime.respond("添加三项待办", on_event=on_event)
            assert visible == ["我先添加。", "共有三项。", "三项", "均已添加。"]
            assert [todo["title"] for todo in list_todos(runtime.connection)] == executed
            assert result.content == "三项均已添加。"
            assert result.messages[2]["content"] == "我先添加。共有三项。"
            assert result.usage.input_tokens == 30
            assert result.usage.output_tokens == 12
            # 第二次请求没有返回缓存明细，不能把第一次的已知值冒充整轮总量。
            assert result.usage.cached_input_tokens is None
            assert result.last_request_usage.input_tokens == 20
            assert len(result.tool_results) == 3
            assert "text_delta" not in [event["event"] for event in result.events]
    assert all(body.closed for body in bodies)


@pytest.mark.parametrize("error", [KeyboardInterrupt(), OSError("连接中断")])
@pytest.mark.parametrize("after_tool", [False, True])
def test_interruption_discards_partial_but_retains_completed_tools(tmp_path, error, after_tool):
    class Model:
        calls = 0

        def stream(self, *, messages, tools, on_text):
            self.calls += 1
            if self.calls == 1 and after_tool:
                return {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "add_1",
                            "function": {"name": "add_todo", "arguments": '{"title":"保留这项"}'},
                        }
                    ],
                }
            on_text("未完成正文")
            raise error

    with create_runtime(model=Model(), db_path=tmp_path / "todos.sqlite3") as runtime:
        visible = []
        with pytest.raises(type(error)):
            runtime.respond("创建待办", on_event=visible.append)
        assert runtime.last_result is None
        assert all(message.get("content") != "未完成正文" for message in runtime.history())
        if after_tool:
            assert [m["role"] for m in runtime.history()] == ["system", "user", "assistant", "tool"]
            assert list_todos(runtime.connection)[0]["title"] == "保留这项"
        else:
            assert [m["role"] for m in runtime.history()] == ["system"]
            assert list_todos(runtime.connection) == []
        assert visible[-1]["event"] == "model_error"


def test_interrupt_during_tool_batch_keeps_ids_paired_and_does_not_run_remaining(tmp_path):
    calls = []

    def interrupt():
        calls.append("interrupt")
        raise KeyboardInterrupt

    response = {
        "content": None,
        "tool_calls": [
            {"id": str(i), "function": {"name": name, "arguments": "{}"}}
            for i, name in enumerate(["first", "interrupt", "never"])
        ],
    }
    model = SimpleNamespace(complete=lambda **_: response)
    with create_runtime(model=model, db_path=tmp_path / "todos.sqlite3") as runtime:
        runtime.tool_registry.handlers.update(
            {
                "first": lambda: calls.append("first") or {"ok": True},
                "interrupt": interrupt,
                "never": lambda: calls.append("never"),
            }
        )
        with pytest.raises(KeyboardInterrupt):
            runtime.respond("执行三个工具", on_event=lambda _: None)
        assert calls == ["first", "interrupt"]
        tool_messages = [m for m in runtime.history() if m["role"] == "tool"]
        assert [m["tool_call_id"] for m in tool_messages] == ["0", "1", "2"]
        assert json.loads(tool_messages[0]["content"]) == {"ok": True}
        assert "可能已产生副作用" in tool_messages[1]["content"]
        assert "未执行" in tool_messages[2]["content"]


def test_fallback_complete_model_delivers_intermediate_and_final_text(tmp_path):
    responses = iter(
        [
            {
                "content": "先查询。",
                "tool_calls": [
                    {"id": "read_1", "function": {"name": "list_todos", "arguments": "{}"}},
                ],
            },
            {"content": "没有待办。", "tool_calls": []},
        ]
    )
    model = SimpleNamespace(complete=lambda **_: next(responses))
    with create_runtime(model=model, db_path=tmp_path / "todos.sqlite3") as runtime:
        events = []
        result = runtime.respond("查询待办", on_event=events.append)
    assert [e["text"] for e in events if e["event"] == "text_delta"] == ["先查询。", "没有待办。"]
    assert result.content == "没有待办。"


@pytest.mark.parametrize("error", [KeyboardInterrupt(), OSError("日志写入失败")])
def test_failure_while_reporting_tool_result_keeps_remaining_calls_paired(tmp_path, error):
    calls = []
    response = {
        "content": None,
        "tool_calls": [{"id": str(i), "function": {"name": "demo", "arguments": "{}"}} for i in range(3)],
    }
    responses = iter([response, {"content": "已检查历史。", "tool_calls": []}])
    model = SimpleNamespace(complete=lambda **_: next(responses))
    with create_runtime(model=model, db_path=tmp_path / "todos.sqlite3") as runtime:
        runtime.tool_registry.handlers["demo"] = lambda: calls.append("demo") or {"ok": True}

        def fail(event):
            if event["event"] == "tool_result":
                raise error

        runtime.logger = fail
        with pytest.raises(type(error)):
            runtime.respond("执行三次", on_event=lambda _: None)
        runtime.logger = None
        tool_messages = [m for m in runtime.history() if m["role"] == "tool"]
        assert [m["tool_call_id"] for m in tool_messages] == ["0", "1", "2"]
        assert json.loads(tool_messages[0]["content"]) == {"ok": True}
        assert all("未执行" in m["content"] for m in tool_messages[1:])
        assert calls == ["demo"]
        assert runtime.respond("查询状态", on_event=lambda _: None).content == "已检查历史。"


@pytest.mark.parametrize("reason,last_arguments", [(None, "{}"), ("length", "{}"), ("tool_calls", '{"title":')])
def test_incomplete_stream_never_executes_even_already_complete_tool_calls(tmp_path, reason, last_arguments):
    body = SSEBody(
        [
            delta(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "first",
                            "type": "function",
                            "function": {"name": "add_todo", "arguments": '{"title":"不应执行"}'},
                        },
                        {
                            "index": 1,
                            "id": "last",
                            "type": "function",
                            "function": {"name": "add_todo", "arguments": last_arguments},
                        },
                    ]
                }
            ),
            delta(reason=reason),
        ]
    )
    transport = httpx2.MockTransport(
        lambda _: httpx2.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            stream=body,
        )
    )
    with OpenAI(api_key="test", max_retries=0, http_client=httpx2.Client(transport=transport)) as client:
        with create_runtime(
            model=OpenAICompatibleModel(model="test-model", client=client), db_path=tmp_path / "todos.sqlite3"
        ) as runtime:
            with pytest.raises(ValueError):
                runtime.respond("添加待办", on_event=lambda _: None)
            assert list_todos(runtime.connection) == []
            assert len(runtime.history()) == 1
    assert body.closed


def test_anthropic_sdk_stream_replays_signed_content_through_runtime(tmp_path):
    from anthropic import Anthropic

    from kairos.providers import AnthropicMessagesModel

    def start():
        return {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "test",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 1, "cache_read_input_tokens": 2},
            },
        }

    def end(reason):
        return [
            {
                "type": "message_delta",
                "delta": {"stop_reason": reason, "stop_sequence": None},
                "usage": {"output_tokens": 5},
            },
            {"type": "message_stop"},
        ]

    frames = [start()]
    blocks = [
        {"type": "thinking", "thinking": "先查待办", "signature": "test-signature"},
        {"type": "text", "text": "我先查询。"},
        {"type": "tool_use", "id": "query_1", "name": "list_todos", "input": {}},
    ]
    for i, block in enumerate(blocks):
        frames += [
            {"type": "content_block_start", "index": i, "content_block": block},
            {"type": "content_block_stop", "index": i},
        ]
    first = SSEBody([*frames, *end("tool_use")], anthropic=True)
    second = SSEBody(
        [
            start(),
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "没有待办。"}},
            {"type": "content_block_stop", "index": 0},
            *end("end_turn"),
        ],
        anthropic=True,
    )
    requests = []

    def handle(request):
        body = json.loads(request.content)
        if requests:
            assert first.closed and first.drained
            assert body["messages"][1] == {"role": "assistant", "content": blocks}
            assert body["messages"][2]["content"] == [
                {
                    "type": "tool_result",
                    "tool_use_id": "query_1",
                    "content": "[]",
                }
            ]
        requests.append(body)
        return httpx2.Response(
            200, headers={"Content-Type": "text/event-stream"}, stream=first if len(requests) == 1 else second
        )

    with Anthropic(
        api_key="test", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(handle))
    ) as client:
        with create_runtime(
            model=AnthropicMessagesModel("test", client=client), db_path=tmp_path / "todos.sqlite3"
        ) as runtime:
            events = []
            result = runtime.respond("查询待办", on_event=events.append)
            assert result.content == "没有待办。"
            assert result.messages[2]["provider_content"] == blocks
            assert result.usage.input_tokens == 24
            assert result.usage.cached_input_tokens == 4
            assert [e["text"] for e in events if e["event"] == "text_delta"] == ["我先查询。", "没有待办。"]
    assert first.closed and second.closed
