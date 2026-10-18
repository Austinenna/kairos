"""真实 SDK 配合离线 HTTP transport，验证图文的协议序列化。"""

import json
from io import BytesIO

import httpx2
import pytest
from anthropic import Anthropic
from openai import OpenAI
from PIL import Image

from kairos.context import REFERENCE_HEADER
from kairos.images import prepare_image
from kairos.providers import AnthropicMessagesModel, OpenAICompatibleModel
from kairos.runtime import create_runtime


def picture():
    stream = BytesIO()
    Image.new("RGB", (32, 24), "red").save(stream, format="PNG")
    return prepare_image(stream.getvalue(), name="offline.png")


def openai_reply(*, stream, tool):
    calls = [{"id": "read_1", "type": "function", "function": {"name": "list_todos", "arguments": "{}"}}]
    text = "先查询。" if tool else "看到了图片。"
    reason = "tool_calls" if tool else "stop"
    base = {"id": "chatcmpl_test", "created": 0, "model": "test-model"}
    if not stream:
        message = {"role": "assistant", "content": text, "reasoning_content": "保留推理字段"}
        if tool:
            message["tool_calls"] = calls
        return httpx2.Response(
            200,
            json={
                **base,
                "object": "chat.completion",
                "choices": [{"index": 0, "finish_reason": reason, "message": message}],
            },
        )
    delta = {"content": text, "reasoning_content": "保留推理字段"}
    if tool:
        delta["tool_calls"] = [{"index": 0, **calls[0]}]
    frames = [
        {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]},
    ]
    data = "".join(
        f"data: {json.dumps({**base, 'object': 'chat.completion.chunk', **frame}, ensure_ascii=False)}\n\n"
        for frame in frames
    )
    return httpx2.Response(
        200, headers={"Content-Type": "text/event-stream"}, content=(data + "data: [DONE]\n\n").encode()
    )


def anthropic_reply(*, stream, tool):
    blocks = [{"type": "text", "text": "先查询。" if tool else "看到了图片。"}]
    if tool:
        blocks.insert(0, {"type": "thinking", "thinking": "保留推理块", "signature": "offline-signature"})
        blocks.append({"type": "tool_use", "id": "read_1", "name": "list_todos", "input": {}})
    reason = "tool_use" if tool else "end_turn"
    message = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "test-model",
        "content": blocks,
        "stop_reason": reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }
    if not stream:
        return httpx2.Response(200, json=message)
    events = [{"type": "message_start", "message": {**message, "content": [], "stop_reason": None}}]
    for index, block in enumerate(blocks):
        events.extend(
            [
                {"type": "content_block_start", "index": index, "content_block": block},
                {"type": "content_block_stop", "index": index},
            ]
        )
    events.extend(
        [
            {"type": "message_delta", "delta": {"stop_reason": reason}, "usage": {"output_tokens": 3}},
            {"type": "message_stop"},
        ]
    )
    data = "".join(f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events)
    return httpx2.Response(200, headers={"Content-Type": "text/event-stream"}, content=data.encode())


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
def test_sdk_sends_full_image_in_both_modes_and_retains_it_across_tools_and_followup(tmp_path, provider, stream):
    image = picture()
    requests = []
    events = []

    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert bool(payload.get("stream")) is stream
        reply = openai_reply if provider == "openai" else anthropic_reply
        return reply(stream=stream, tool=len(requests) == 1)

    client_class = OpenAI if provider == "openai" else Anthropic
    with client_class(
        api_key="offline-test", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(handle))
    ) as client:
        model = (
            OpenAICompatibleModel(model="test-model", client=client)
            if provider == "openai"
            else AnthropicMessagesModel("test-model", client=client)
        )
        with create_runtime(model=model, db_path=tmp_path / "test.sqlite3") as runtime:
            result = runtime.respond("结合图片查询待办", images=[image], on_event=events.append if stream else None)
            assert result.content == "看到了图片。"
            runtime.respond("继续解释", on_event=events.append if stream else None)
            assert runtime.messages[1]["content"][-1]["data"] == image.data
            assert image.data not in json.dumps(result.events + events)

    assert len(requests) == 3
    if provider == "openai":
        expected = {"type": "image_url", "image_url": {"url": f"data:{image.mime_type};base64,{image.data}"}}
    else:
        expected = {"type": "image", "source": {"type": "base64", "media_type": image.mime_type, "data": image.data}}
    for request in requests:
        references = [
            message for message in request["messages"] if str(message.get("content", "")).startswith(REFERENCE_HEADER)
        ]
        assert len(references) == 1
        user = next(
            message
            for message in request["messages"]
            if message["role"] == "user"
            and isinstance(message["content"], list)
            and message["content"][0] == {"type": "text", "text": "结合图片查询待办"}
        )
        assert user["content"] == [
            {"type": "text", "text": "结合图片查询待办"},
            {"type": "text", "text": "[图片 1：offline.png]"},
            expected,
        ]
    assert requests[-1]["messages"][-1] == {"role": "user", "content": "继续解释"}
    if provider == "openai":
        assistant = next(message for message in requests[1]["messages"] if message["role"] == "assistant")
        assert assistant["reasoning_content"] == "保留推理字段"
        assert assistant["tool_calls"][0]["function"]["name"] == "list_todos"
        result = next(message for message in requests[1]["messages"] if message["role"] == "tool")
        assert result == {"role": "tool", "tool_call_id": "read_1", "content": "[]"}
        assert assistant["tool_calls"][0]["id"] == result["tool_call_id"]
    else:
        assistant = next(message for message in requests[1]["messages"] if message["role"] == "assistant")
        assert assistant["content"][0] == {
            "type": "thinking",
            "thinking": "保留推理块",
            "signature": "offline-signature",
        }
        result = next(
            message
            for message in requests[1]["messages"]
            if isinstance(message["content"], list) and message["content"][0].get("type") == "tool_result"
        )
        assert result == {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "read_1", "content": "[]"}],
        }
        assert assistant["content"][-1]["id"] == result["content"][0]["tool_use_id"]
