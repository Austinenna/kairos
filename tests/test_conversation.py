import json

import pytest

from kairos import conversation as conversation_module
from kairos.config import Settings
from kairos.conversation import Conversation
from kairos.images.prepare import Image
from kairos.models import ModelResponse, Usage
from kairos.services import open_services


class FakeModel:
    model = "deepseek-flash"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def stream(self, messages, tools, on_text=None):
        self.requests.append(messages)
        return self.responses.pop(0)


@pytest.fixture
def services(tmp_path):
    settings = Settings(_env_file=None, kairos_model="deepseek-flash", gateway_base_url="http://gateway")
    services = open_services(settings, tmp_path, auto_memory=False)
    yield services
    services.close()


@pytest.fixture
def new(services, monkeypatch):
    """新建一个会话，模型换成按顺序返回给定响应的 FakeModel。"""

    def new(*responses):
        model = FakeModel(*responses)
        monkeypatch.setattr(conversation_module, "build_model", lambda *args, **kwargs: model)
        return Conversation.create(services), model

    return new


def test_turn_is_saved_and_can_be_resumed(services, new):
    call = {"id": "c1", "type": "function", "function": {"name": "add_todo", "arguments": '{"title": "买牛奶"}'}}
    conversation, _ = new(
        ModelResponse(content="", tool_calls=[call], usage=Usage(100, 80, 10)),
        ModelResponse(content="已添加", usage=Usage(120, 100, 5)),
    )

    conversation.respond("记一下买牛奶")

    store = services.sessions
    resumed = store.load_messages(conversation.session.id)
    assert [m["role"] for m in resumed] == ["system", "user", "assistant", "tool", "assistant"]
    assert resumed == conversation.messages
    assert "买牛奶" in resumed[3]["content"]  # 真的调用了待办工具
    assert store.get(conversation.session.id[:8]).title == "记一下买牛奶"
    totals = conversation.totals()
    assert (totals.usage.input_tokens, totals.usage.cached_input_tokens, totals.requests) == (220, 180, 2)
    assert totals.cost_usd > 0


def test_request_view_adds_rules_memory_and_time_without_saving_them(services, new):
    services.memory.add_core("用户叫 Austin")
    conversation, model = new(ModelResponse(content="你好 Austin"))

    conversation.respond("你好")

    request = model.requests[0]
    assert request[0]["content"] == conversation.messages[0]["content"]  # 开头是会话自己的提示词
    assert "记忆使用规则" in request[1]["content"]  # 紧接着是固定的工具规则
    assert "用户叫 Austin" in request[-1]["content"] and "当前时间" in request[-1]["content"]  # 末尾是会变的内容
    saved = json.dumps(services.sessions.load_messages(conversation.session.id), ensure_ascii=False)
    assert "Austin" not in saved.replace("你好 Austin", "")


def test_empty_sessions_are_not_listed(services, new):
    new()
    used, _ = new(ModelResponse(content="你好"))
    used.respond("你好")

    assert [s.id for s in services.sessions.recent()] == [used.session.id]


def test_images_are_sent_and_history_tools_see_only_text(services, new):
    conversation, model = new(ModelResponse(content="是一只猫"))
    image = Image(name="cat.png", mime_type="image/png", data="QUJD", width=1, height=1)

    conversation.respond("这是什么", [image])

    assert model.requests[0][-2]["content"][1] == image.to_block()  # 倒数第二条是用户消息，最后是时间
    assert services.sessions.get(conversation.session.id).title == "这是什么 [图片：cat.png]"
    page = services.tools.handlers["read_session"](conversation.session.id)
    assert page["messages"][0]["content"] == "这是什么 [图片：cat.png]"  # 不含 base64
