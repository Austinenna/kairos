import json

import pytest

from kairos import conversation as conversation_module
from kairos.config import Settings
from kairos.conversation import Conversation
from kairos.models import ModelResponse, Usage
from kairos.storage import SessionStore, open_db


class FakeModel:
    model = "deepseek-flash"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def stream(self, messages, tools, on_text=None):
        self.requests.append(messages)
        return self.responses.pop(0)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """返回 (store, 新建会话的函数)。模型换成 FakeModel，数据库在临时目录。"""
    store = SessionStore(open_db(tmp_path / "kairos.db"))
    settings = Settings(_env_file=None, kairos_model="deepseek-flash", gateway_base_url="http://gateway")

    def new(*responses):
        model = FakeModel(*responses)
        monkeypatch.setattr(conversation_module, "build_model", lambda *args, **kwargs: model)
        handlers = {"add_todo": lambda title: {"id": 1, "title": title}}
        return Conversation.create(store, settings, [], handlers), model

    return store, new


def test_turn_is_saved_and_can_be_resumed(setup):
    store, new = setup
    call = {"id": "c1", "type": "function", "function": {"name": "add_todo", "arguments": '{"title": "买牛奶"}'}}
    conversation, _ = new(
        ModelResponse(content="", tool_calls=[call], usage=Usage(100, 80, 10)),
        ModelResponse(content="已添加", usage=Usage(120, 100, 5)),
    )

    conversation.respond("记一下买牛奶")

    resumed = store.load_messages(conversation.session.id)
    assert [m["role"] for m in resumed] == ["system", "user", "assistant", "tool", "assistant"]
    assert resumed == conversation.messages
    assert store.get(conversation.session.id[:8]).title == "记一下买牛奶"
    totals = conversation.totals()
    assert (totals.usage.input_tokens, totals.usage.cached_input_tokens, totals.requests) == (220, 180, 2)
    assert totals.cost_usd > 0


def test_time_note_is_sent_but_not_saved(setup):
    store, new = setup
    conversation, model = new(ModelResponse(content="今天是星期四"))

    conversation.respond("今天星期几")

    request = model.requests[0]
    assert request[-1]["role"] == "system" and "当前时间" in request[-1]["content"]
    saved = store.load_messages(conversation.session.id)
    assert not any("当前时间" in json.dumps(m, ensure_ascii=False) for m in saved)


def test_empty_sessions_are_not_listed(setup):
    store, new = setup
    new()
    used, _ = new(ModelResponse(content="你好"))
    used.respond("你好")

    assert [s.id for s in store.recent()] == [used.session.id]
