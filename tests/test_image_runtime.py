"""图片在输入、历史和观察边界间的离线回归验证。"""

import copy
import json
from io import BytesIO

import pytest
from PIL import Image

from kairos.images import ImageInputError, load_image, prepare_image
from kairos.observability import JsonlLogger
from kairos.runtime import create_runtime
from kairos.todos import list_todos


def image_bytes():
    stream = BytesIO()
    Image.new("RGB", (40, 30), (12, 34, 56)).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def picture():
    return prepare_image(image_bytes(), name="截图.png")


def tool_response():
    return {
        "content": "我会添加截图中的事项。",
        "tool_calls": [
            {"id": "add_1", "type": "function", "function": {"name": "add_todo", "arguments": '{"title":"读图待办"}'}},
        ],
    }


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, *, messages, tools):
        self.requests.append(copy.deepcopy(messages))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


def image_blocks(messages):
    return [
        block
        for message in messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "image"
    ]


def test_image_snapshot_survives_file_changes_tool_round_and_plain_followup(tmp_path):
    path = tmp_path / "截图.png"
    path.write_bytes(image_bytes())
    picture = load_image(path)
    path.write_bytes(b"the original file has changed")
    model = FakeModel([tool_response(), {"content": "已添加。"}, {"content": "继续解释原图。"}])
    events = []
    log_path = tmp_path / "events.jsonl"
    with create_runtime(model=model, db_path=tmp_path / "test.sqlite3", log_path=log_path) as runtime:
        first = runtime.respond("按截图添加待办", images=[picture], on_event=events.append)
        assert runtime.last_input_retained
        second = runtime.respond("再解释一下", on_event=events.append)
        assert runtime.last_input_retained
        assert len(model.requests) == 3
        assert all(image_blocks(request)[0]["data"] == picture.data for request in model.requests)
        assert image_blocks(first.messages)[0]["data"] == picture.data
        assert image_blocks(second.messages)[0]["data"] == picture.data
        assert model.requests[-1][-1] == {"role": "user", "content": "再解释一下"}
        assert list_todos(runtime.connection)[0]["title"] == "读图待办"
        visible_history = runtime.history()
        assert picture.data not in json.dumps(visible_history)
        visible_history[1]["content"].clear()
        assert image_blocks(runtime.messages)[0]["data"] == picture.data
        assert picture.data not in json.dumps(first.events + second.events + events)
        assert picture.data not in log_path.read_text()
        runtime.reset()
        assert not runtime.last_input_retained


def test_image_only_prompt_and_empty_text_validation(tmp_path, picture):
    model = FakeModel([{"content": "一张图片。"}])
    with create_runtime(model=model, db_path=tmp_path / "test.sqlite3") as runtime:
        runtime.respond("  ", images=[picture])
        assert model.requests[0][1]["content"][0] == {"type": "text", "text": "请描述这些图片。"}
        before = copy.deepcopy(runtime.messages)
        with pytest.raises(ValueError, match="不能为空"):
            runtime.respond("  ")
        assert runtime.messages == before
        assert not runtime.last_input_retained


def test_budget_rejection_does_not_change_history_or_call_model(tmp_path, picture):
    model = FakeModel([{"content": "收到。"}])
    with create_runtime(model=model, db_path=tmp_path / "test.sqlite3") as runtime:
        runtime.respond("已有消息")
        before = copy.deepcopy(runtime.messages)
        with pytest.raises(ImageInputError):
            runtime.check_image_budget([picture] * 1000)
        with pytest.raises(ImageInputError):
            runtime.respond("附件过多", images=[picture] * 1000)
        assert runtime.messages == before
        assert len(model.requests) == 1
        assert not runtime.last_input_retained
        assert runtime.last_result is None


def test_stable_input_labels_reach_model_without_renumbering(tmp_path, picture):
    model = FakeModel([{"content": "收到。"}])
    with create_runtime(model=model, db_path=tmp_path / "test.sqlite3") as runtime:
        runtime.respond("看图片 7", images=[picture], image_labels=["图片 7"])
        texts = [b["text"] for b in model.requests[0][-1]["content"] if b["type"] == "text"]
        assert any("[图片 7：" in text for text in texts)
        assert not any("[图片 1：" in text for text in texts)


@pytest.mark.parametrize("after_tool", [False, True])
@pytest.mark.parametrize("error", [OSError("连接中断"), KeyboardInterrupt()])
def test_failure_retains_image_input_only_when_completed_tool_history_exists(tmp_path, picture, after_tool, error):
    model = FakeModel(([tool_response()] if after_tool else []) + [error])
    with create_runtime(model=model, db_path=tmp_path / "test.sqlite3") as runtime:
        with pytest.raises(type(error)):
            runtime.respond("按图添加", images=[picture], on_event=lambda _: None)
        assert runtime.last_input_retained is after_tool
        assert bool(image_blocks(runtime.messages)) is after_tool
        assert len(list_todos(runtime.connection)) == int(after_tool)
        assert runtime.last_result is None


def test_error_and_observation_callbacks_cannot_expose_or_mutate_image_payload(tmp_path, picture):
    model = FakeModel([RuntimeError(f"failed: data:{picture.mime_type};base64,{picture.data}; raw={picture.data}")])
    events = []
    log_path = tmp_path / "error.jsonl"

    def observe(event):
        events.append(copy.deepcopy(event))
        if event["event"] == "model_request":
            event["messages"][1]["content"].clear()

    with create_runtime(model=model, db_path=tmp_path / "test.sqlite3", log_path=log_path) as runtime:
        with pytest.raises(RuntimeError) as raised:
            runtime.respond("解释图片", images=[picture], on_event=observe)
        assert picture.data not in str(raised.value)
        assert picture.data not in json.dumps(events)
        assert picture.data not in log_path.read_text()
        assert image_blocks(model.requests[0])[0]["data"] == picture.data
        assert not runtime.last_input_retained
        assert len(runtime.messages) == 1


def test_closed_runtime_clears_last_input_retained(tmp_path, picture):
    runtime = create_runtime(model=FakeModel([{"content": "收到"}]), db_path=tmp_path / "test.sqlite3")
    runtime.respond("解释图片", images=[picture])
    assert runtime.last_input_retained
    runtime.close()
    with pytest.raises(RuntimeError, match="已经关闭"):
        runtime.respond("继续")
    assert not runtime.last_input_retained


def test_jsonl_logger_defensively_redacts_direct_image_events(tmp_path, picture):
    path = tmp_path / "direct.jsonl"
    event = {"event": "debug", "content": [picture.to_block()]}
    JsonlLogger(path)(event)
    assert picture.data not in path.read_text()
    assert event["content"][0]["data"] == picture.data
