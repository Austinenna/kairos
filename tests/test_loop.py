import json

from kairos.agent import run_turn
from kairos.models import ModelResponse


class FakeModel:
    """按顺序返回预先写好的响应，并记下每次收到的请求。"""

    model = "fake"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def stream(self, messages, tools, on_text=None):
        self.requests.append(list(messages))
        response = self.responses.pop(0)
        if on_text and response.content:
            on_text(response.content)
        return response


def call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def test_tool_result_goes_back_to_model_before_final_answer():
    model = FakeModel(
        ModelResponse(content="", tool_calls=[call("c1", "add", {"a": 1, "b": 2})]),
        ModelResponse(content="结果是 3"),
    )
    messages = [{"role": "user", "content": "1+2"}]
    saved = []

    result = run_turn(messages, model, [], {"add": lambda a, b: a + b}, on_message=saved.append)

    assert result.content == "结果是 3"
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[2] == {"role": "tool", "tool_call_id": "c1", "content": "3"}
    # 第二次请求时，模型能看到工具结果
    assert model.requests[1][-1]["role"] == "tool"
    assert saved == messages[1:]


def test_tool_errors_are_returned_to_model():
    def broken():
        raise ValueError("坏了")

    model = FakeModel(
        ModelResponse(content="", tool_calls=[call("c1", "broken", {}), call("c2", "missing", {})]),
        ModelResponse(content="抱歉"),
    )
    messages = [{"role": "user", "content": "试试"}]

    run_turn(messages, model, [], {"broken": broken})

    assert "ValueError: 坏了" in json.loads(messages[2]["content"])["error"]
    assert "没有名为 missing 的工具" in json.loads(messages[3]["content"])["error"]
