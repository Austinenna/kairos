import json
from io import BytesIO

import pytest
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from test_archive import FilingModel, jpeg_bytes

from kairos import conversation as conversation_module
from kairos import services as services_module
from kairos.config import Settings
from kairos.models import ModelResponse
from kairos.services import open_services
from kairos.wechat import api
from kairos.wechat import runner as runner_module
from kairos.wechat.api import Credentials, Incoming, WeixinClient
from kairos.wechat.runner import WeixinRunner, split_reply


class FakeModel:
    model = "deepseek-flash"

    def stream(self, messages, tools, on_text=None):
        return ModelResponse(content=f"收到：{messages[-2]['content']}")  # 最后一条是时间，倒数第二条是用户消息


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setattr(conversation_module, "build_model", lambda *args, **kwargs: FakeModel())
    filing = {"category": "票据", "title": "停车票", "description": "停车 12 元"}
    monkeypatch.setattr(services_module, "build_model", lambda *args, **kwargs: FilingModel(filing))
    settings = Settings(_env_file=None, kairos_model="deepseek-flash", gateway_base_url="http://gateway")
    services = open_services(settings, tmp_path, auto_memory=False)
    account = {"bot_token": "t", "bot_id": "bot", "owner_id": "me"}
    (tmp_path / "account.json").write_text(json.dumps(account))
    runner = WeixinRunner(services, tmp_path)
    runner.sent = []
    runner.client.send = lambda text, token: runner.sent.append((text, token))
    runner.client.download_image = lambda image_item: jpeg_bytes()
    yield runner
    services.close()


def test_message_is_answered_and_session_survives_restart(runner, tmp_path):
    runner._handle(Incoming("你好", [], "token-1"))
    runner._handle(Incoming("/new", [], "token-2"))
    runner._handle(Incoming("第二个会话", [], "token-3"))

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
        {"from_user_id": "me", "context_token": "d", "item_list": [{"type": 2, "image_item": {"aeskey": "k"}}]},
        {"from_user_id": "me", "context_token": "e", "item_list": [{"type": 3}]},  # 语音，不支持
    ]
    monkeypatch.setattr(api, "request", lambda *args, **kwargs: {"msgs": messages, "get_updates_buf": "next"})

    incoming, cursor = WeixinClient(Credentials("t", "bot", "me")).poll("old")

    assert [(m.text, m.images, m.context_token) for m in incoming] == [
        ("你好", [], "a"),
        ("", [{"aeskey": "k"}], "d"),
        (None, [], "e"),
    ]
    assert cursor == "next"


def test_long_reply_is_split_by_bytes():
    parts = split_reply("一" * 1000 + "\n" + "二" * 10, limit=1800)

    assert "".join(parts) == "一" * 1000 + "\n" + "二" * 10
    assert all(len(part.encode()) <= 1800 for part in parts)


def test_image_is_archived_and_receipt_sent(runner):
    runner._handle(Incoming("", [{"aeskey": "k"}], "token-1"))

    [(receipt, token)] = runner.sent
    assert receipt.startswith("已存到 票据/") and receipt.endswith("-停车票.jpg\n停车 12 元")
    assert token == "token-1"
    assert runner.conversation is None  # 只发图片不进对话


def test_download_image_decrypts_cdn_data(monkeypatch):
    key = bytes(range(16))
    padder = padding.PKCS7(128).padder()
    padded = padder.update(b"original image") + padder.finalize()
    encryptor = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    encrypted = encryptor.update(padded) + encryptor.finalize()
    monkeypatch.setattr(api.urllib.request, "urlopen", lambda url, timeout: BytesIO(encrypted))

    image_item = {"aeskey": key.hex(), "media": {"full_url": "https://cdn/download"}}
    assert WeixinClient(Credentials("t", "bot", "me")).download_image(image_item) == b"original image"


def test_images_go_into_the_next_message_unless_too_old(runner, monkeypatch):
    runner._handle(Incoming("", [{"aeskey": "k"}], "t1"))
    runner._handle(Incoming("", [{"aeskey": "k"}], "t2"))
    runner._handle(Incoming("这两张是什么", [], "t3"))
    runner._handle(Incoming("", [{"aeskey": "k"}], "t4"))
    runner.last_image_at -= runner_module.PAIR_SECONDS + 1  # 发完图过了很久才说话
    runner._handle(Incoming("明天提醒我开会", [], "t5"))

    users = [m for m in runner.conversation.messages if m["role"] == "user"]
    assert [b["image_id"] for b in users[0]["content"] if b["type"] == "image_ref"] == [1, 2]
    assert users[1]["content"] == "明天提醒我开会"
    assert runner.pending_images == []
