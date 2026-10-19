"""用隔离状态和假网络验证微信探针，不连接微信或读取个人配置。"""

import importlib.util
import io
import json
import stat
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit

import pytest


@pytest.fixture(scope="module")
def probe():
    """加载独立探针模块供离线测试调用。

    作用:
        检查脚本能在未启动登录或网络流程时安全导入。

    参数:
        无。

    返回:
        已加载的探针模块。

    行为说明:
        1. 只加载项目脚本，不运行命令行入口。
    """
    path = Path(__file__).resolve().parents[1] / "scripts" / "weixin_probe.py"
    spec = importlib.util.spec_from_file_location("kairos_weixin_probe_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def state():
    """提供无真实凭据的初始探针状态。

    作用:
        为持久化及消息处理用例提供固定身份和空发送记录。

    参数:
        无。

    返回:
        每个测试独立使用的状态字典。

    行为说明:
        1. 凭据和身份均为合成数据。
    """
    return {
        "credentials": {
            "bot_token": "synthetic-private-token",
            "bot_id": "bot-test",
            "owner_id": "owner-test",
            "base_url": "https://ilinkai.weixin.qq.com",
        },
        "cursor": "cursor-before",
        "replies": {},
    }


@pytest.mark.parametrize("url", ["https://ilinkai.weixin.qq.com", "https://ilinkai.weixin.qq.com/"])
def test_accepts_https_weixin_root(probe, url):
    """接受微信 HTTPS 根地址。

    作用:
        检查正常默认服务地址可用于探针。

    参数:
        probe: 探针模块。
        url: 有无末尾斜杠的正常地址。

    返回:
        None。

    行为说明:
        1. 校验规范化结果仍为同一安全根地址。
    """
    parsed = urlsplit(probe.validate_base_url(url))
    assert parsed.scheme == "https"
    assert parsed.hostname == "ilinkai.weixin.qq.com"
    assert parsed.path in {"", "/"}
    assert not parsed.query and not parsed.fragment


@pytest.mark.parametrize(
    "url",
    [
        "http://ilinkai.weixin.qq.com",
        "https://ilinkai.weixin.qq.com.attacker.invalid",
        "https://weixin.qq.com.attacker.invalid",
        "https://attacker.invalid/weixin.qq.com",
        "https://user:secret@ilinkai.weixin.qq.com",
        "https://ilinkai.weixin.qq.com/another-path",
        "https://ilinkai.weixin.qq.com/?token=secret",
        "https://ilinkai.weixin.qq.com/#fragment",
        "file:///tmp/weixin.qq.com",
        "//ilinkai.weixin.qq.com",
    ],
)
def test_rejects_unsafe_service_urls(probe, url):
    """拒绝可能把凭据发送到其他目标的服务地址。

    作用:
        限制登录返回地址和恢复状态中的请求目标。

    参数:
        probe: 探针模块。
        url: 非 HTTPS、冒充域名或携带额外内容的地址。

    返回:
        None。

    行为说明:
        1. 不发请求，只检查地址准入。
    """
    with pytest.raises(probe.ProbeError):
        probe.validate_base_url(url)


def test_private_state_round_trip_preserves_large_integer(probe, state, tmp_path):
    """验证状态私有保存并精确保留大整数。

    作用:
        保证重启恢复不会丢失凭据、游标或发送记录中的消息 ID。

    参数:
        probe: 探针模块。
        state: 合成状态。
        tmp_path: 隔离目录。

    返回:
        None。

    行为说明:
        1. 创建私有目录及状态文件，再读取比浮点安全整数更大的 ID。
    """
    path = tmp_path / "private-probe" / "state.json"
    state["test_message_id"] = 2**63 - 1
    probe.save_state(path, state)
    assert probe.load_state(path) == state
    assert json.loads(path.read_text())["test_message_id"] == 2**63 - 1
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_state_replacement_keeps_private_permissions(probe, state, tmp_path):
    """验证替换旧状态时恢复文件私有权限。

    作用:
        防止已有状态文件权限过宽时继续暴露登录凭据。

    参数:
        probe: 探针模块。
        state: 合成状态。
        tmp_path: 隔离目录。

    返回:
        None。

    行为说明:
        1. 替换原有宽权限文件并核对内容和权限。
    """
    path = tmp_path / "state.json"
    path.write_text('{"old":true}')
    path.chmod(0o644)
    probe.save_state(path, state)
    assert probe.load_state(path) == state
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


class FakeClient:
    """记录发送请求并允许注入发送前检查或传输失败。"""

    def __init__(self, *, before_request=None, failure=None, response=None):
        """建立不执行网络操作的客户端。

        作用:
            供消息处理测试观察发送内容和持久化顺序。

        参数:
            before_request: 发送时执行的可选检查。
            failure: 需要模拟的异常。
            response: 需要覆盖的响应字典。

        返回:
            None。

        行为说明:
            1. 只保存测试配置。
        """
        self.calls = []
        self.before_request = before_request
        self.failure = failure
        self.response = {"ret": 0, "errcode": 0} if response is None else response

    def request(self, endpoint, payload=None, timeout=35.0):
        """记录一次假请求并返回成功结果或模拟失败。

        作用:
            代替真实微信发送边界。

        参数:
            endpoint: 请求端点。
            payload: 请求 JSON。
            timeout: 透传的等待上限。

        返回:
            成功状态字典。

        异常:
            注入的异常: 配置 failure 时抛出。

        行为说明:
            1. 先记录请求和执行检查，再决定返回或失败。
        """
        self.calls.append((endpoint, payload, timeout))
        if self.before_request is not None:
            self.before_request()
        if self.failure is not None:
            raise self.failure
        return self.response


def incoming(text="Kairos 探针 TEST-123", message_id=2**63 - 1, **overrides):
    """构造微信私聊文字消息。

    作用:
        为身份过滤、精确匹配和幂等测试提供合成载荷。

    参数:
        text: 消息文字。
        message_id: 微信消息标识。
        overrides: 需要覆盖的消息字段。

    返回:
        独立消息字典。

    行为说明:
        1. 默认构造本人发给机器人的完整文字消息。
    """
    return {
        "message_id": message_id,
        "from_user_id": "owner-test",
        "to_user_id": "bot-test",
        "message_type": 1,
        "message_state": 2,
        "context_token": "synthetic-context-token",
        "item_list": [{"type": 1, "text_item": {"text": text}}],
        **overrides,
    }


def test_matched_message_persists_before_send_and_survives_restart(probe, state, tmp_path):
    """验证先保存发送状态再回复，重启后不重复发送。

    作用:
        检查消息接收、发送和本地持久化之间的关键边界。

    参数:
        probe: 探针模块。
        state: 合成状态。
        tmp_path: 隔离目录。

    返回:
        None。

    行为说明:
        1. 在假发送内部确认 sending 已落盘。
        2. 成功后重新加载状态并重投原消息，确认没有再次发送。
    """
    path = tmp_path / "state.json"
    expected = "Kairos 探针 TEST-123"
    batch = {"msgs": [incoming(expected)], "get_updates_buf": "cursor-after"}

    def check_sending():
        """在传输边界读取已持久化的发送标记。

        作用:
            确认程序不是先发消息再写去重状态。

        参数:
            无。

        返回:
            None。

        行为说明:
            1. 从文件重新读取，不依赖内存状态。
        """
        stored = probe.load_state(path)
        assert len(stored["replies"]) == 1
        assert next(iter(stored["replies"].values()))["status"] == "sending"

    client = FakeClient(before_request=check_sending)
    assert probe.process_batch(client, state, path, batch, expected) is True
    assert len(client.calls) == 1
    endpoint, payload, _ = client.calls[0]
    assert endpoint.endswith("/sendmessage")
    assert payload["msg"]["to_user_id"] == "owner-test"
    assert payload["msg"]["context_token"] == "synthetic-context-token"
    assert payload["msg"]["item_list"] == [{"type": 1, "text_item": {"text": probe.REPLY_TEXT}}]
    restored = probe.load_state(path)
    assert restored["cursor"] == "cursor-after"
    key, record = next(iter(restored["replies"].items()))
    assert str(2**63 - 1) not in key
    assert record["status"] == "sent"
    assert record["acknowledgement"] == "business"
    assert expected not in path.read_text()
    assert "synthetic-context-token" not in path.read_text()
    duplicate_client = FakeClient()
    probe.process_batch(duplicate_client, restored, path, batch, expected)
    assert duplicate_client.calls == []


def test_failed_send_becomes_uncertain_and_is_not_retried(probe, state, tmp_path):
    """验证发送异常后保留未知状态并拒绝自动重发。

    作用:
        避免服务器可能已收到消息时，重启或重复投递导致再次回复。

    参数:
        probe: 探针模块。
        state: 合成状态。
        tmp_path: 隔离目录。

    返回:
        None。

    行为说明:
        1. 模拟传输失败，读取 uncertain 记录并重新处理同一消息。
    """
    path = tmp_path / "state.json"
    expected = "Kairos 探针 TEST-123"
    batch = {"msgs": [incoming(expected)], "get_updates_buf": "cursor-after"}
    client = FakeClient(failure=probe.ProbeError("模拟传输结果未知"))
    with pytest.raises(probe.ProbeError):
        probe.process_batch(client, state, path, batch, expected)
    restored = probe.load_state(path)
    assert len(client.calls) == 1
    assert next(iter(restored["replies"].values()))["status"] == "uncertain"
    assert restored["cursor"] == "cursor-before"
    duplicate_client = FakeClient()
    probe.process_batch(duplicate_client, restored, path, batch, expected)
    assert duplicate_client.calls == []


@pytest.mark.parametrize(
    "message",
    [
        incoming(from_user_id="someone-else"),
        incoming(from_user_id="bot-test", to_user_id="owner-test", message_type=2),
        incoming(message_type=2),
        incoming(message_type=0),
        incoming(to_user_id="another-bot"),
        incoming(group_id="test-group"),
        incoming("其他消息"),
        incoming("Kairos 探针 TEST-123\n附加文字"),
        incoming(" Kairos 探针 TEST-123 "),
        incoming(item_list=[{"type": 2, "image_item": {}}]),
    ],
)
def test_unrelated_or_incomplete_messages_never_reply(probe, state, tmp_path, message):
    """验证非本人、非测试文字和缺少关键字段的消息不回复。

    作用:
        将探针的实际发送限制在已授权的本人测试消息内。

    参数:
        probe: 探针模块。
        state: 合成状态。
        tmp_path: 隔离目录。
        message: 应忽略的输入消息。

    返回:
        None。

    行为说明:
        1. 处理输入并确认既无发送也无成功回复记录。
    """
    path = tmp_path / "state.json"
    client = FakeClient()
    batch = {"msgs": [message], "get_updates_buf": "cursor-after"}
    assert probe.process_batch(client, state, path, batch, "Kairos 探针 TEST-123") is False
    assert client.calls == []
    assert probe.load_state(path)["replies"] == {}


@pytest.mark.parametrize("message", [incoming(context_token=""), incoming(message_id=""), incoming(message_id=True)])
def test_matched_message_without_usable_delivery_identity_stops_safely(probe, state, tmp_path, message):
    """验证命中口令但无法安全回复时停止且没有发送。

    作用:
        将缺少消息标识或回复凭据的输入保留为待检查状态。

    参数:
        probe: 探针模块。
        state: 合成状态。
        tmp_path: 隔离目录。
        message: 缺少有效关联字段的匹配消息。

    返回:
        None。

    行为说明:
        1. 允许程序明确停止，但禁止发送或推进游标。
    """
    client = FakeClient()
    with pytest.raises(probe.ProbeError):
        probe.process_batch(client, state, tmp_path / "state.json", {"msgs": [message]}, "Kairos 探针 TEST-123")
    assert client.calls == []
    assert state["replies"] == {}
    assert state["cursor"] == "cursor-before"


def test_single_batch_only_replies_once(probe, state, tmp_path):
    """验证一个探针批次最多发送一条固定回复。

    作用:
        避免批次内重复或多次发送口令放大探针行为。

    参数:
        probe: 探针模块。
        state: 合成状态。
        tmp_path: 隔离目录。

    返回:
        None。

    行为说明:
        1. 同一批中放入重复 ID 及另一个匹配 ID，确认总共只发送一次。
    """
    client = FakeClient()
    messages = [incoming(), incoming(), incoming(message_id=2**63)]
    assert probe.process_batch(client, state, tmp_path / "state.json", {"msgs": messages}, "Kairos 探针 TEST-123")
    assert len(client.calls) == 1


@pytest.mark.parametrize("status", ["sending", "uncertain", "submitted", "sent"])
def test_persisted_delivery_state_is_not_replayed(probe, state, tmp_path, status):
    """验证已有发送状态跨重启后不会触发重发。

    作用:
        覆盖发送开始、结果未知、HTTP 接受和业务确认后的重启边界。

    参数:
        probe: 探针模块。
        state: 合成状态。
        tmp_path: 隔离目录。
        status: 需要恢复的发送状态。

    返回:
        None。

    行为说明:
        1. 将已知消息的持久记录设为指定状态，再模拟重启后重复投递。
    """
    path = tmp_path / "state.json"
    batch = {"msgs": [incoming()]}
    probe.process_batch(FakeClient(), state, path, batch, "Kairos 探针 TEST-123")
    next(iter(state["replies"].values()))["status"] = status
    probe.save_state(path, state)
    client = FakeClient()
    probe.process_batch(client, probe.load_state(path), path, batch, "Kairos 探针 TEST-123")
    assert client.calls == []


@pytest.mark.parametrize(
    ("response", "status", "acknowledgement"),
    [
        ({}, "submitted", "http_only"),
        ({"ret": None}, "submitted", "http_only"),
        ({"errcode": None}, "submitted", "http_only"),
        ({"ret": None, "errcode": None}, "submitted", "http_only"),
        ({"ret": 0}, "sent", "business"),
        ({"errcode": 0}, "sent", "business"),
        ({"ret": 0, "errcode": 0}, "sent", "business"),
        ({"ret": None, "errcode": 0}, "sent", "business"),
        ({"ret": 0, "errcode": None}, "sent", "business"),
    ],
)
def test_optional_success_codes_record_acknowledgement(probe, state, tmp_path, response, status, acknowledgement):
    """验证可选成功码缺失时完成提交，并保留确认依据。

    作用:
        区分 HTTP 请求成功返回和微信显式业务成功码，不把缺失字段误判为失败。

    参数:
        probe: 探针模块。
        state: 合成状态。
        tmp_path: 隔离目录。
        response: 缺少错误码、空错误码或含数值零的合成响应。
        status: 应保存的发送状态。
        acknowledgement: 应保存的确认依据。

    返回:
        None。

    行为说明:
        1. 各种成功返回均结束本次探针；缺少业务码时只记录 HTTP 接受。
        2. 重启后重新处理原消息不重复发送。
    """
    path = tmp_path / "state.json"
    client = FakeClient(response=response)
    batch = {"msgs": [incoming()], "get_updates_buf": "cursor-after"}
    assert probe.process_batch(client, state, path, batch, "Kairos 探针 TEST-123") is True
    restored = probe.load_state(path)
    record = next(iter(restored["replies"].values()))
    assert record["status"] == status
    assert record["acknowledgement"] == acknowledgement
    assert restored["cursor"] == "cursor-after"
    duplicate_client = FakeClient()
    probe.process_batch(duplicate_client, restored, path, batch, "Kairos 探针 TEST-123")
    assert duplicate_client.calls == []


@pytest.mark.parametrize(
    "response", [{"ret": -1}, {"errcode": 401}, {"ret": 0, "errcode": 401}, {"ret": -1, "errcode": 0}]
)
def test_nonzero_business_codes_stop_without_replay(probe, state, tmp_path, response):
    """验证任一非零业务码都停止并保留结果未知状态。

    作用:
        防止只检查一个成功字段而掩盖另一个字段报告的失败。

    参数:
        probe: 探针模块。
        state: 合成状态。
        tmp_path: 隔离目录。
        response: 至少包含一个非零错误码的合成响应。

    返回:
        None。

    行为说明:
        1. 发送失败保留原游标；重启收到原消息时不自动重试。
    """
    path = tmp_path / "state.json"
    client = FakeClient(response=response)
    batch = {"msgs": [incoming()], "get_updates_buf": "cursor-after"}
    with pytest.raises(probe.ProbeError):
        probe.process_batch(client, state, path, batch, "Kairos 探针 TEST-123")
    restored = probe.load_state(path)
    assert next(iter(restored["replies"].values()))["status"] == "uncertain"
    assert restored["cursor"] == "cursor-before"
    duplicate_client = FakeClient()
    probe.process_batch(duplicate_client, restored, path, batch, "Kairos 探针 TEST-123")
    assert duplicate_client.calls == []


def test_atomic_write_failure_preserves_previous_state(probe, state, tmp_path, monkeypatch):
    """验证替换失败时旧状态完整保留且临时文件被清理。

    作用:
        防止磁盘写入中断损坏已保存登录状态。

    参数:
        probe: 探针模块。
        state: 合成状态。
        tmp_path: 隔离目录。
        monkeypatch: 局部替换原子替换操作。

    返回:
        None。

    行为说明:
        1. 模拟最后替换失败，核对原始状态和目录残留。
    """
    path = tmp_path / "state.json"
    probe.save_state(path, state)
    original = path.read_bytes()

    def fail_replace(*args):
        """模拟操作系统原子替换失败。

        作用:
            验证替换前的已写临时文件会被清理。
        参数:
            args: 原始替换调用参数。
        返回:
            无。
        异常:
            OSError: 始终抛出模拟故障。
        行为说明:
            1. 不执行实际文件替换。
        """
        raise OSError("模拟替换失败")

    monkeypatch.setattr(probe.os, "replace", fail_replace)
    state["cursor"] = "new-cursor"
    with pytest.raises(OSError):
        probe.save_state(path, state)
    assert path.read_bytes() == original
    assert list(tmp_path.glob(".state-*")) == []


def test_http_methods_headers_and_redirect_blocking(probe, monkeypatch):
    """验证请求方法、认证边界和重定向禁用。

    作用:
        检查登录查询不携带 Bearer，发送请求不会自动跨域转发。

    参数:
        probe: 探针模块。
        monkeypatch: 替换客户端网络边界。

    返回:
        None。

    行为说明:
        1. 用内存响应记录 GET 与 POST，再直接验证安装的重定向处理器。
    """
    client = probe.ProbeClient(token="synthetic-private-token")
    requests = []

    def open_request(request, timeout):
        """记录 urllib 请求并返回内存 JSON。

        作用:
            阻止本测试接触网络。
        参数:
            request: 待检查的 urllib 请求。
            timeout: 请求等待上限。
        返回:
            可作为响应使用的内存字节流。
        行为说明:
            1. 只记录参数，不发送请求。
        """
        requests.append((request, timeout))
        return io.BytesIO(b'{"ret":0}')

    monkeypatch.setattr(client.opener, "open", open_request)
    assert client.request("/ilink/bot/get_qrcode_status?test=1") == {"ret": 0}
    assert client.request("/ilink/bot/sendmessage", {"text": "合成消息"}, timeout=15.0) == {"ret": 0}
    get, post = requests[0][0], requests[1][0]
    assert get.get_method() == "GET" and get.get_header("Authorization") is None
    assert post.get_method() == "POST"
    assert post.get_header("Authorization") == "Bearer synthetic-private-token"
    assert json.loads(post.data) == {"text": "合成消息"}
    assert requests[1][1] == 15.0
    handlers = [handler for handler in client.opener.handlers if isinstance(handler, probe.NoRedirect)]
    assert len(handlers) == 1
    assert handlers[0].redirect_request(post, None, 302, "redirect", {}, "https://attacker.invalid") is None


@pytest.mark.parametrize("kind", ["http", "ret", "errcode", "string_code", "invalid_json"])
def test_network_errors_never_expose_response_or_credentials(probe, monkeypatch, kind):
    """验证 HTTP 与业务错误只展示安全信息。

    作用:
        防止响应正文、URL 或字符串错误码中的凭据出现在异常展示中。

    参数:
        probe: 探针模块。
        monkeypatch: 替换网络边界。
        kind: 错误类型。

    返回:
        None。

    行为说明:
        1. 构造携带合成秘密的失败响应，确认用户可见异常不包含该内容。
    """
    secret = "DO-NOT-EXPOSE-SYNTHETIC-SECRET"
    client = probe.ProbeClient(token=secret)
    bodies = {
        "ret": json.dumps({"ret": -7, "errmsg": secret}).encode(),
        "errcode": json.dumps({"errcode": 401, "errmsg": secret}).encode(),
        "string_code": json.dumps({"errcode": secret, "errmsg": secret}).encode(),
        "invalid_json": secret.encode(),
    }

    def open_request(request, timeout):
        """返回带合成秘密的错误响应。

        作用:
            在内存中模拟传输和业务层拒绝。
        参数:
            request: 未发送的请求。
            timeout: 未使用的等待上限。
        返回:
            包含错误 JSON 的字节流。
        异常:
            HTTPError: kind 为 http 时抛出模拟 HTTP 错误。
        行为说明:
            1. 不读取真实凭据或发网络请求。
        """
        if kind == "http":
            raise HTTPError("https://example.invalid/?secret=" + secret, 403, secret, {}, io.BytesIO(secret.encode()))
        return io.BytesIO(bodies[kind])

    monkeypatch.setattr(client.opener, "open", open_request)
    with pytest.raises(probe.ProbeError) as caught:
        client.request("/ilink/bot/sendmessage", {"synthetic": secret})
    message = str(caught.value)
    assert secret not in message
    assert "example.invalid" not in message
    if kind == "http":
        assert "403" in message
    elif kind == "ret":
        assert "-7" in message
    elif kind == "errcode":
        assert "401" in message
