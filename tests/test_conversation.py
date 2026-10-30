import json

import pytest

from kairos import conversation as conversation_module
from kairos.compaction import compact as compact_module
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


@pytest.fixture
def compact_early(services, monkeypatch):
    """阈值调成负数、只保留最近一条用户消息起的原文，第三轮开始就会压缩。"""
    services.settings.kairos_auto_compact_window_percent = 1
    monkeypatch.setattr(compact_module, "KEEP_RECENT_TOKENS", 1)


def test_auto_compaction_summarizes_older_turns_and_keeps_recent_ones(services, new, compact_early):
    conversation, model = new(
        ModelResponse(content="答一", usage=Usage(100, 0, 10)),
        ModelResponse(content="答二", usage=Usage(200, 0, 10)),
        ModelResponse(content="摘要正文", finish_reason="stop"),
        ModelResponse(content="答三", usage=Usage(50, 0, 5)),
    )
    notices = []
    conversation.respond("第一")
    conversation.respond("第二")  # 只有一轮可以压缩时不压缩：至少保留最近一整轮
    conversation.respond("第三", on_notice=notices.append)

    summary_request = model.requests[2][1]["content"]
    assert "[用户] 第一" in summary_request and "第二" not in summary_request
    request = model.requests[3]
    assert request[2]["role"] == "user" and "摘要正文" in request[2]["content"]
    assert [m["content"] for m in request[3:-1]] == ["第二", "答二", "第三"]
    assert notices[-1].startswith("已压缩上下文")
    assert len(services.sessions.load_messages(conversation.session.id)) == 7  # 原文一条都没删
    assert conversation.context_tokens() == 55  # 用压缩之后那次请求的用量


def test_manual_compaction_ignores_stale_usage_and_forgets_read_files(services, new, compact_early):
    conversation, _ = new(
        ModelResponse(content="答一", usage=Usage(900_000, 0, 10)),
        ModelResponse(content="答二", usage=Usage(900_000, 0, 10)),
        ModelResponse(content="摘要正文", finish_reason="stop"),
    )
    services.settings.kairos_auto_compact_window_percent = 100
    conversation.respond("第一")
    conversation.respond("第二")
    file_tools = conversation.file_tools

    result = conversation.compact(focus="待办")

    assert result.tokens_before == 900_010
    assert conversation.context_tokens() < 1000  # 压缩前的用量已经过时，改为整段估算
    assert conversation.file_tools is not file_tools  # “已读”记录清空了


def test_failed_compaction_keeps_history_and_still_answers(services, new, compact_early):
    conversation, model = new(
        ModelResponse(content="答一"),
        ModelResponse(content="答二"),
        ModelResponse(content="摘要写到一半", finish_reason="length"),
        ModelResponse(content="答三"),
    )
    notices = []
    conversation.respond("第一")
    conversation.respond("第二")

    assert conversation.respond("第三", on_notice=notices.append).content == "答三"
    assert "压缩失败" in notices[-1]
    assert services.sessions.latest_compaction(conversation.session.id) is None
    assert model.requests[3][2]["content"] == "第一"  # 这一轮用的是完整历史
