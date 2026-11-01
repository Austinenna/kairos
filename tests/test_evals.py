import json

import pytest

from kairos import conversation as conversation_module
from kairos.config import Settings
from kairos.conversation import Conversation
from kairos.evals.cases import save_case
from kairos.evals.trace import load_trace
from kairos.models import ModelResponse, Usage
from kairos.services import open_services


class FakeModel:
    model = "deepseek-flash"

    def __init__(self, *responses):
        self.responses = list(responses)

    def stream(self, messages, tools, on_text=None):
        return self.responses.pop(0)


@pytest.fixture
def services(tmp_path):
    settings = Settings(_env_file=None, kairos_model="deepseek-flash", gateway_base_url="http://gateway")
    services = open_services(settings, tmp_path, auto_memory=False)
    yield services
    services.close()


@pytest.fixture
def conversation(services, monkeypatch):
    """一个已经聊了两轮的会话：第一轮直接回答；第二轮调了一个成功和一个不存在的工具，再回答。"""
    ok = {"id": "c1", "type": "function", "function": {"name": "add_todo", "arguments": '{"title": "买牛奶"}'}}
    bad = {"id": "c2", "type": "function", "function": {"name": "no_such_tool", "arguments": "{}"}}
    model = FakeModel(
        ModelResponse(content="你好", usage=Usage(100, 0, 5)),
        ModelResponse(content="我来记一下", tool_calls=[ok, bad], usage=Usage(200, 100, 10)),
        ModelResponse(content="已添加买牛奶", usage=Usage(300, 200, 8)),
    )
    monkeypatch.setattr(conversation_module, "build_model", lambda *args, **kwargs: model)
    conversation = Conversation.create(services)
    conversation.respond("你好")
    conversation.respond("记一下买牛奶")
    return conversation


def test_trace_lists_requests_tools_and_errors(services, conversation):
    lines = load_trace(services.sessions, conversation.session.id, "", width=160)

    texts = [line.text for line in lines]
    assert "2 次请求" in texts[0] and "输入 500（缓存 300）输出 18" in texts[0]
    assert texts[1] == "用户：记一下买牛奶"
    assert texts[2].startswith("第 1 次请求  输入 200（缓存 100）输出 10")
    assert texts[3] == "  说明：我来记一下"
    assert texts[4].startswith('  → add_todo {"title": "买牛奶"}')
    assert texts[5].startswith("    ← ") and not lines[5].error
    assert texts[6].startswith("  → no_such_tool")
    assert lines[7].error and "没有名为 no_such_tool 的工具" in texts[7]
    assert texts[8].startswith("第 2 次请求") and texts[9] == "  回答：已添加买牛奶"


def test_trace_can_look_back_and_at_another_session(services, conversation):
    other = Conversation.create(services)

    lines = load_trace(services.sessions, other.session.id, f"2 {conversation.session.id[:8]}", width=160)

    assert lines[1].text == "用户：你好" and lines[-1].text == "  回答：你好"
    with pytest.raises(ValueError):
        load_trace(services.sessions, other.session.id, "nosuchsession", width=160)


def test_bad_saves_latest_turn_with_its_context(services, conversation, isolated_app_dir):
    path = save_case(services.sessions, conversation.session, "工具报错了还说已添加", "用户叫 Austin")

    assert path.parent == isolated_app_dir / "evals" / "cases"
    case = json.loads(path.read_text(encoding="utf-8"))
    assert case["note"] == "工具报错了还说已添加" and case["expected"] == ""
    assert [m["role"] for m in case["context"]] == ["system", "user", "assistant"]  # 这一轮之前的历史
    assert case["input"]["content"] == "记一下买牛奶"
    assert [step["message"]["role"] for step in case["trace"]] == ["assistant", "tool", "tool", "assistant"]
    assert case["trace"][0]["usage"]["input_tokens"] == 200
