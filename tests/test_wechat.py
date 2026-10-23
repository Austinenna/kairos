import json

import pytest

from kairos import conversation as conversation_module
from kairos.config import Settings
from kairos.models import ModelResponse
from kairos.services import open_services
from kairos.wechat import api
from kairos.wechat.api import Credentials, WeixinClient
from kairos.wechat.runner import WeixinRunner, split_reply


class FakeModel:
    model = "deepseek-flash"

    def stream(self, messages, tools, on_text=None):
        return ModelResponse(content=f"收到：{messages[-2]['content']}")  # 最后一条是时间，倒数第二条是用户消息


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setattr(conversation_module, "build_model", lambda *args, **kwargs: FakeModel())
    settings = Settings(_env_file=None, kairos_model="deepseek-flash", gateway_base_url="http://gateway")
    services = open_services(settings, tmp_path, auto_memory=False)
    account = {"bot_token": "t", "bot_id": "bot", "owner_id": "me"}
    (tmp_path / "account.json").write_text(json.dumps(account))
    runner = WeixinRunner(services, tmp_path)
    runner.sent = []
    runner.client.send = lambda text, token: runner.sent.append((text, token))
    yield runner
    services.close()


def test_message_is_answered_and_session_survives_restart(runner, tmp_path):
    runner._handle("你好", "token-1")
    runner._handle("/new", "token-2")
    runner._handle("第二个会话", "token-3")

    assert runner.sent == [("收到：你好", "token-1"), ("已开始新会话。", "token-2"), ("收到：第二个会话", "token-3")]
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["session_id"] == runner.conversation.session.id  # 重启后接着用 /new 之后的会话
    session = runner.services.sessions.get(state["session_id"])
    assert (session.title, session.channel) == ("第二个会话", "weixin")


def test_poll_keeps_only_complete_messages_from_owner(monkeypatch):
    messages = [
        {"from_user_id": "me", "context_token": "a", "item_list": [{"type": 1, "text_item": {"text": "你好"}}]},
        {"from_user_id": "someone", "context_token": "b", "item_list": [{"type": 1, "text_item": {"text": "x"}}]},
        {"from_user_id": "me", "context_token": "c", "message_state": 1, "item_list": []},  # 还在输入中
        {"from_user_id": "me", "context_token": "d", "item_list": [{"type": 2}]},  # 图片
    ]
    monkeypatch.setattr(api, "request", lambda *args, **kwargs: {"msgs": messages, "get_updates_buf": "next"})

    incoming, cursor = WeixinClient(Credentials("t", "bot", "me")).poll("old")

    assert [(m.text, m.context_token) for m in incoming] == [("你好", "a"), (None, "d")]
    assert cursor == "next"


def test_long_reply_is_split_by_bytes():
    parts = split_reply("一" * 1000 + "\n" + "二" * 10, limit=1800)

    assert "".join(parts) == "一" * 1000 + "\n" + "二" * 10
    assert all(len(part.encode()) <= 1800 for part in parts)
