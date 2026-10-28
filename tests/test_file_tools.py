import json

from kairos import conversation as conversation_module
from kairos.config import Settings
from kairos.conversation import Conversation
from kairos.models import ModelResponse
from kairos.services import open_services


class FakeModel:
    model = "deepseek-flash"

    def __init__(self, *responses):
        self.responses = list(responses)

    def stream(self, messages, tools, on_text=None):
        return self.responses.pop(0)


def _call(call_id: str, name: str, **arguments) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def test_model_must_read_before_editing(tmp_path, isolated_workspace, monkeypatch):
    (isolated_workspace / "待办.md").parent.mkdir(parents=True)
    (isolated_workspace / "待办.md").write_text("- 周五体检\n")
    edit = {"path": "待办.md", "old_string": "周五", "new_string": "周六"}
    model = FakeModel(
        ModelResponse(content="", tool_calls=[_call("c1", "edit_file", **edit)]),
        ModelResponse(content="", tool_calls=[_call("c2", "read_file", path="待办.md", offset=None, limit=None)]),
        ModelResponse(content="", tool_calls=[_call("c3", "edit_file", **edit)]),
        ModelResponse(content="改好了"),
    )
    monkeypatch.setattr(conversation_module, "build_model", lambda *args, **kwargs: model)
    settings = Settings(_env_file=None, kairos_model="deepseek-flash", gateway_base_url="http://gateway")
    services = open_services(settings, tmp_path, auto_memory=False)

    conversation = Conversation.create(services)
    conversation.respond("体检改到周六")

    results = [json.loads(m["content"]) for m in conversation.messages if m["role"] == "tool"]
    assert "还没有读过" in results[0]["error"]
    assert results[1]["content"] == "1\t- 周五体检"  # offset、limit 传 null 时用默认值
    assert (isolated_workspace / "待办.md").read_text() == "- 周六体检\n"
    services.close()
