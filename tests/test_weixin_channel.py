"""使用合成身份、假 HTTPS 与隔离目录验证微信正式通道。"""

import copy
import fcntl
import http.client
import io
import json
import stat
import sys
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from kairos.channels import weixin
from kairos.channels.base import ReplyTarget, TransportError


@pytest.fixture
def credentials():
    """创建合成的本人账号凭据。

    作用:
        避免测试读取本机实际账号。
    参数:
        无。
    返回:
        每次测试独立的凭据字典。
    行为说明:
        1. 所有标识与令牌均为测试数据。
    """
    return {
        "bot_token": "synthetic-secret-token",
        "bot_id": "test-bot",
        "owner_id": "test-owner",
        "base_url": weixin.DEFAULT_BASE,
    }


class FakeClient:
    """提供预置响应并记录请求，不访问网络。"""

    def __init__(self, *results):
        """保存顺序返回的结果。

        作用:
            为适配器和登录状态机模拟传输。
        参数:
            results: 响应对象或异常。
        返回:
            无返回值。
        行为说明:
            1. 独立保存调用记录。
        """
        self.results = list(results)
        self.calls = []

    def request(self, endpoint, payload=None, timeout=35.0):
        """记录一次请求并消耗预置结果。

        作用:
            验证真实发起的协议参数与请求次数。
        参数:
            endpoint, payload, timeout: 待记录的请求参数。
        返回:
            预置响应对象。
        异常:
            Exception: 当前预置结果为异常时原样抛出。
        行为说明:
            1. 不处理实际网络请求。
        """
        self.calls.append((endpoint, payload, timeout))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def message(**changes):
    """构造可覆盖字段的本人纯文字消息。

    作用:
        为正常路径及异常输入提供同一基线。
    参数:
        changes: 需要替换的协议字段。
    返回:
        合成消息字典。
    行为说明:
        1. 消息 ID 大于 JavaScript 的精确整数范围。
    """
    return {
        "message_id": 2**63 - 1,
        "from_user_id": "test-owner",
        "to_user_id": "test-bot",
        "message_type": 1,
        "message_state": 2,
        "context_token": "synthetic-context",
        "item_list": [{"type": 1, "text_item": {"text": "继续之前的任务"}}],
        **changes,
    }


def test_poll_preserves_ids_and_keeps_protocol_state_out_of_messages(credentials):
    """验证本人消息及游标令牌一起返回，原状态不被修改。

    作用:
        确保协调层可以原子保存输入与游标，模型不会收到微信凭据。
    参数:
        credentials: 合成凭据。
    返回:
        None。
    行为说明:
        1. 核对大整数、普通文本、请求游标和新 checkpoint。
    """
    client = FakeClient({"ret": 0, "msgs": [message()], "get_updates_buf": "next"})
    adapter = weixin.WeixinAdapter(credentials, client)
    checkpoint = {"cursor": "old", "context_tokens": {"test-owner": "old-context"}}
    previous = copy.deepcopy(checkpoint)
    batch = adapter.poll(checkpoint)
    assert checkpoint == previous
    assert len(batch.messages) == 1
    inbound = batch.messages[0]
    assert inbound.message_id == str(2**63 - 1)
    assert inbound.text == "继续之前的任务"
    assert inbound.conversation_id == inbound.sender_id == adapter.owner_id
    assert not hasattr(inbound, "context_token")
    assert batch.checkpoint == {"cursor": "next", "context_tokens": {"test-owner": "synthetic-context"}}
    assert client.calls[0][1]["get_updates_buf"] == "old"


@pytest.mark.parametrize(
    "changes",
    [
        {"from_user_id": "someone-else"},
        {"message_type": 2},
        {"group_id": "some-group"},
        {"to_user_id": "another-bot"},
        {"message_state": 1},
        {"item_list": [{"type": 1, "is_completed": False, "text_item": {"text": "片段"}}]},
    ],
)
def test_poll_ignores_other_targets_and_partial_input(credentials, changes):
    """过滤非本人私聊及尚未完成的输入。

    作用:
        保证没有回复其他人或提前执行片段的入口。
    参数:
        credentials: 合成凭据。
        changes: 不应被执行的消息变体。
    返回:
        None。
    行为说明:
        1. 被过滤消息不会更新本人回复令牌。
    """
    adapter = weixin.WeixinAdapter(credentials, FakeClient({"msgs": [message(**changes)], "get_updates_buf": "next"}))
    batch = adapter.poll({})
    assert batch.messages == ()
    assert batch.checkpoint == {"cursor": "next", "context_tokens": {}}


@pytest.mark.parametrize(
    "items,expected",
    [
        ([{"type": 2, "image_item": {}}], None),
        ([{"type": 1, "text_item": {"text": "看图"}}, {"type": 2}], None),
        (
            [{"type": 1, "text_item": {"text": "第一行"}}, {"type": 1, "text_item": {"text": "第二行"}}],
            "第一行\n第二行",
        ),
        ([{"type": 1, "text_item": None}], None),
        ([], None),
    ],
)
def test_poll_returns_unsupported_message_without_model_text(credentials, items, expected):
    """保留媒体输入身份但不伪造可交给模型的文本。

    作用:
        让协调层统一记录并回复暂不支持的消息。
    参数:
        credentials: 合成凭据。
        items: 纯文字、媒体或混合内容。
        expected: 允许传递给模型的文本。
    返回:
        None。
    行为说明:
        1. 同时检查省略 type/state 的官方兼容情况。
    """
    inbound = message(item_list=items)
    del inbound["message_type"]
    del inbound["message_state"]
    batch = weixin.WeixinAdapter(credentials, FakeClient({"msgs": [inbound]})).poll({})
    assert batch.messages[0].text == expected


@pytest.mark.parametrize(
    "changes", [{"message_id": None}, {"message_id": True}, {"message_id": -1}, {"context_token": ""}]
)
def test_poll_rejects_owner_message_missing_delivery_identity(credentials, changes):
    """阻止缺少可靠身份或回复令牌的本人消息推进游标。

    作用:
        避免错误格式造成无法去重或无法回传的任务。
    参数:
        credentials: 合成凭据。
        changes: 无效协议字段。
    返回:
        None。
    行为说明:
        1. 失败后调用方的已持久化 checkpoint 不变化。
    """
    checkpoint = {"cursor": "old"}
    adapter = weixin.WeixinAdapter(credentials, FakeClient({"msgs": [message(**changes)], "get_updates_buf": "next"}))
    with pytest.raises(TransportError):
        adapter.poll(checkpoint)
    assert checkpoint == {"cursor": "old"}


def test_poll_timeout_preserves_checkpoint(credentials):
    """将正常长轮询超时转换为空批次。

    作用:
        允许协调层继续等待，不误报任务执行失败。
    参数:
        credentials: 合成凭据。
    返回:
        None。
    行为说明:
        1. 保留原有游标与令牌，不重置同步位置。
    """
    checkpoint = {"cursor": "old", "context_tokens": {"test-owner": "old-context"}}
    adapter = weixin.WeixinAdapter(credentials, FakeClient(weixin.WeixinTimeout("等待超时")))
    batch = adapter.poll(checkpoint)
    assert not batch.messages
    assert batch.checkpoint == checkpoint
    assert batch.checkpoint is not checkpoint


@pytest.mark.parametrize(
    "response,acknowledged",
    [({"ret": 0}, True), ({"errcode": 0}, True), ({"ret": 0.0}, True), ({}, False), ({"ret": False}, False)],
)
def test_send_uses_persisted_target_and_distinguishes_business_ack(credentials, response, acknowledged):
    """核对发送目标、令牌、唯一 ID 与业务确认等级。

    作用:
        防止空成功响应被误当作手机收件回执。
    参数:
        credentials: 合成凭据。
        response: 微信接口的不同响应。
        acknowledged: 是否存在明确数值零业务码。
    返回:
        None。
    行为说明:
        1. 每次调用只发送一次，不重新生成 client_id。
    """
    client = FakeClient(response)
    adapter = weixin.WeixinAdapter(credentials, client)
    receipt = adapter.send(
        ReplyTarget("weixin", "test-bot", "test-owner"),
        "答案",
        "stable-client-id",
        {"context_tokens": {"test-owner": "ct"}},
    )
    assert receipt.acknowledged is acknowledged
    assert len(client.calls) == 1
    msg = client.calls[0][1]["msg"]
    assert msg["to_user_id"] == "test-owner"
    assert msg["context_token"] == "ct"
    assert msg["client_id"] == "stable-client-id"
    assert msg["message_type"] == msg["message_state"] == 2


@pytest.mark.parametrize(
    "target",
    [
        ReplyTarget("other", "test-bot", "test-owner"),
        ReplyTarget("weixin", "other", "test-owner"),
        ReplyTarget("weixin", "test-bot", "other"),
    ],
)
def test_send_rejects_cross_account_targets_before_network(credentials, target):
    """拒绝跨账号、跨通道或跨用户发送。

    作用:
        即使协调层传错路由也不会发送给他人。
    参数:
        credentials: 合成凭据。
        target: 非当前本人对话的目标。
    返回:
        None。
    行为说明:
        1. 确认拒绝发生在任何网络调用之前。
    """
    client = FakeClient()
    with pytest.raises(TransportError):
        weixin.WeixinAdapter(credentials, client).send(target, "答案", "id", {})
    assert not client.calls


@pytest.mark.parametrize(
    "response", [{"ret": -14}, {"ret": -2, "errcode": -14}, {"ret": -2, "errmsg": "synthetic-secret"}]
)
def test_business_errors_are_redacted_and_authentication_takes_priority(credentials, response):
    """检查认证失效优先级及错误脱敏。

    作用:
        让协调层立即停止失效账号，并避免输出服务端原文。
    参数:
        credentials: 合成凭据。
        response: 认证或普通业务错误。
    返回:
        None。
    行为说明:
        1. 不复制 errmsg，任何 -14 均归类为认证问题。
    """
    adapter = weixin.WeixinAdapter(credentials, FakeClient(response))
    with pytest.raises(TransportError) as caught:
        adapter.poll({})
    assert "synthetic-secret" not in str(caught.value)
    assert caught.value.authentication is (-14 in response.values())


class FakeOpener:
    """模拟 urllib 响应，允许检查头及注入网络异常。"""

    def __init__(self, result):
        """保存待返回字节或异常。

        作用:
            隔离真实 HTTPS 请求。
        参数:
            result: 字节或异常。
        返回:
            无返回值。
        行为说明:
            1. 初始没有请求记录。
        """
        self.result = result
        self.requests = []

    def open(self, request, timeout):
        """记录请求并返回字节流。

        作用:
            检查真实网络封装的错误边界。
        参数:
            request: urllib 请求对象。
            timeout: 等待上限。
        返回:
            可作为上下文管理器的 BytesIO。
        异常:
            Exception: 预置结果为异常时抛出。
        行为说明:
            1. 不连接网络。
        """
        self.requests.append(request)
        if isinstance(self.result, Exception):
            raise self.result
        return io.BytesIO(self.result)


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("synthetic-secret"),
        URLError("synthetic-secret"),
        OSError("synthetic-secret"),
        http.client.IncompleteRead(b"synthetic-secret"),
    ],
)
def test_send_network_failure_is_uncertain_and_not_retried(credentials, failure):
    """保留发送网络失败的未知结果，禁止内部重试。

    作用:
        防止网络超时触发重复回复或重复工具执行。
    参数:
        credentials: 合成凭据。
        failure: 不同传输错误。
    返回:
        None。
    行为说明:
        1. 核对异常标记与唯一一次请求，并检查原始错误不泄漏。
    """
    client = weixin.WeixinClient(credentials["base_url"], credentials["bot_token"])
    client.opener = FakeOpener(failure)
    adapter = weixin.WeixinAdapter(credentials, client)
    with pytest.raises(TransportError) as caught:
        adapter.send(
            ReplyTarget("weixin", "test-bot", "test-owner"), "答案", "id", {"context_tokens": {"test-owner": "ct"}}
        )
    assert caught.value.uncertain and not caught.value.retryable
    assert "synthetic-secret" not in str(caught.value)
    assert len(client.opener.requests) == 1


@pytest.mark.parametrize("raw", [b"", b"{}", b'{"ret":0}'])
def test_client_accepts_empty_http_success_and_uses_post_headers(raw):
    """接受缺少业务码的成功响应并构造协议头。

    作用:
        将接口兼容性和明确业务确认区分开。
    参数:
        raw: 空正文或正常 JSON。
    返回:
        None。
    行为说明:
        1. 仅 Bot POST 携带 Bearer，GET 不携带它。
    """
    client = weixin.WeixinClient(token="synthetic-token")
    opener = FakeOpener(raw)
    client.opener = opener
    client.request("/ilink/bot/sendmessage", {})
    headers = dict(opener.requests[0].header_items())
    assert headers["Authorization"] == "Bearer synthetic-token"
    assert headers["Ilink-app-clientversion"] == "132106"
    client.request("/ilink/bot/get_qrcode_status?qrcode=synthetic")
    assert "Authorization" not in dict(opener.requests[1].header_items())


@pytest.mark.parametrize("response", [b"secret invalid json", b"[]", b"x" * (weixin.MAX_RESPONSE_BYTES + 1)])
def test_client_rejects_unbounded_or_invalid_responses_without_leaking_body(response):
    """限制响应大小及结构，错误不携带正文。

    作用:
        保护日志边界并保持发送结果未知的事实。
    参数:
        response: 无效或超限响应。
    返回:
        None。
    行为说明:
        1. 发送已发起后的格式错误同样标记 uncertain。
    """
    client = weixin.WeixinClient()
    client.opener = FakeOpener(response)
    with pytest.raises(TransportError) as caught:
        client.request("/ilink/bot/sendmessage", {})
    assert caught.value.uncertain
    assert "secret" not in str(caught.value)


def test_client_rejects_http_redirect_and_reports_safe_status():
    """拒绝 HTTP 重定向而不转交凭据。

    作用:
        校验登录和发送共同使用的主机边界。
    参数:
        无。
    返回:
        None。
    行为说明:
        1. handler 不创建新请求，HTTP 错误仅展示状态码。
    """
    assert weixin.NoRedirect().redirect_request(None, None, 302, "", {}, "https://attacker.invalid") is None
    client = weixin.WeixinClient()
    client.opener = FakeOpener(HTTPError("https://secret.invalid", 302, "secret", {}, None))
    with pytest.raises(TransportError, match="302") as caught:
        client.request("/ilink/bot/getupdates", {})
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize(
    "url",
    [
        "http://ilinkai.weixin.qq.com",
        "https://evil.invalid",
        "https://ilinkai.weixin.qq.com.evil.invalid",
        "https://u:p@ilinkai.weixin.qq.com",
        "https://ilinkai.weixin.qq.com:443",
        "https://ilinkai.weixin.qq.com/path",
        "https://ilinkai.weixin.qq.com/?secret=1",
        "\nhttps://ilinkai.weixin.qq.com",
        "https://[invalid",
    ],
)
def test_rejects_unsafe_api_roots(url):
    """阻止凭据发送到非预期地址。

    作用:
        约束恢复凭据与扫码业务重定向。
    参数:
        url: 非法根地址。
    返回:
        None。
    行为说明:
        1. 不发网络请求，异常不包含地址原文。
    """
    with pytest.raises(TransportError):
        weixin.validate_base_url(url)


def test_credentials_and_probe_import_are_private_atomic_and_recoverable(credentials, tmp_path):
    """验证凭据迁移的私有权限、源文件保留及初始游标恢复。

    作用:
        即使写入账号后初始化 SQLite 前退出，也能保留探针收消息位置。
    参数:
        credentials: 合成凭据。
        tmp_path: 隔离目录。
    返回:
        None。
    行为说明:
        1. 导入不覆盖现有账号，不复制探针回复记录。
    """
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(json.dumps({"credentials": credentials, "cursor": "probe-next", "replies": {"old": {}}}))
    original = probe_path.read_bytes()
    account_path = tmp_path / "formal" / "account.json"
    checkpoint = weixin.import_probe(probe_path, account_path)
    saved = weixin.read_credentials(account_path)
    assert saved == {**credentials, "initial_checkpoint": checkpoint}
    assert checkpoint == {"cursor": "probe-next", "context_tokens": {}}
    assert probe_path.read_bytes() == original
    assert stat.S_IMODE(account_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(account_path.parent.stat().st_mode) == 0o700
    with pytest.raises(TransportError):
        weixin.import_probe(probe_path, account_path)
    assert weixin.read_credentials(account_path) == saved
    weixin.write_credentials(account_path, credentials)
    assert weixin.read_credentials(account_path) == credentials


def test_import_probe_refuses_live_probe_lock(credentials, tmp_path):
    """拒绝与正在运行的独立探针同时拉取账号。

    作用:
        保证迁移使用探针已完成的同步位置。
    参数:
        credentials: 合成凭据。
        tmp_path: 隔离目录。
    返回:
        None。
    行为说明:
        1. 锁被占用时不创建正式账号。
    """
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"credentials": credentials, "cursor": "old"}))
    account = tmp_path / "formal" / "account.json"
    with (tmp_path / ".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(TransportError, match="仍在运行"):
            weixin.import_probe(probe, account)
    assert not account.exists()


@pytest.mark.parametrize("state", [{}, {"credentials": {}}, {"credentials": "secret"}])
def test_import_rejects_missing_fields_before_writing(state, tmp_path):
    """拒绝不完整探针凭据而不创建正式账号。

    作用:
        避免部分导入后丢失既有验证状态。
    参数:
        state: 缺字段的合成状态。
        tmp_path: 隔离目录。
    返回:
        None。
    行为说明:
        1. 不修改源探针文件。
    """
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps(state))
    account = tmp_path / "account.json"
    with pytest.raises(TransportError):
        weixin.import_probe(probe, account)
    assert not account.exists()


def test_login_handles_verification_redirect_and_cleans_qr(credentials, tmp_path, monkeypatch, capsys):
    """验证完整扫码、配对码与业务重定向流程。

    作用:
        确保登录仅保存本人凭据，不输出二维码负载和验证码。
    参数:
        credentials: 合成凭据。
        tmp_path: 隔离目录。
        monkeypatch: 替换网络、二维码库、等待与输入。
        capsys: 捕获安全输出。
    返回:
        None。
    行为说明:
        1. 成功后二维码清除，凭据包含可恢复的空初始 checkpoint。
    """
    qr = {"qrcode": "synthetic-qr-id", "qrcode_img_content": "https://secret-qr.invalid"}
    client = FakeClient(
        qr,
        {"status": "need_verifycode"},
        {"status": "scaned"},
        {"status": "scaned_but_redirect", "redirect_host": "other.weixin.qq.com"},
        {
            "status": "confirmed",
            "bot_token": credentials["bot_token"],
            "ilink_bot_id": credentials["bot_id"],
            "ilink_user_id": credentials["owner_id"],
            "baseurl": credentials["base_url"],
        },
    )
    monkeypatch.setattr(weixin, "WeixinClient", lambda *args: client)
    monkeypatch.setattr(weixin.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(weixin.getpass, "getpass", lambda prompt: "synthetic-verify-code")
    monkeypatch.setitem(
        sys.modules,
        "qrcode",
        SimpleNamespace(make=lambda text: SimpleNamespace(save=lambda stream, **kw: stream.write(b"fake-png"))),
    )
    account = tmp_path / "account.json"
    weixin.login(account)
    assert weixin.read_credentials(account) == {**credentials, "initial_checkpoint": {}}
    assert "verify_code=synthetic-verify-code" in client.calls[2][0]
    assert "verify_code" not in client.calls[3][0]
    assert not list(tmp_path.glob("login-*.png"))
    output = capsys.readouterr().out
    assert "secret-qr" not in output and "synthetic-verify-code" not in output
    assert credentials["bot_token"] not in output


def test_login_refuses_overwrite_before_network(credentials, tmp_path, monkeypatch):
    """在开始扫码前拒绝未经允许的账号覆盖。

    作用:
        避免已有有效凭据被意外替换。
    参数:
        credentials: 合成凭据。
        tmp_path: 隔离目录。
        monkeypatch: 注入无结果的假客户端。
    返回:
        None。
    行为说明:
        1. 已有账号保持逐字节不变。
    """
    account = tmp_path / "account.json"
    weixin.write_credentials(account, credentials)
    original = account.read_bytes()
    client = FakeClient()
    monkeypatch.setattr(weixin, "WeixinClient", lambda: client)
    with pytest.raises(TransportError, match="已存在"):
        weixin.login(account)
    assert not client.calls
    assert account.read_bytes() == original


@pytest.mark.parametrize("failure", ["expired", "verify_code_blocked", "binded_redirect"])
def test_failed_relogin_preserves_existing_credentials_and_removes_qr(credentials, tmp_path, monkeypatch, failure):
    """在重新登录失败时保留旧账号并清除临时二维码。

    作用:
        明确替换授权只在拿到完整新凭据后生效。
    参数:
        credentials: 合成凭据。
        tmp_path: 隔离目录。
        monkeypatch: 替换网络与二维码库。
        failure: 微信拒绝或停止登录的状态。
    返回:
        None。
    行为说明:
        1. 原文件逐字节不变，不残留登录图片。
    """
    account = tmp_path / "account.json"
    weixin.write_credentials(account, credentials)
    original = account.read_bytes()
    client = FakeClient({"qrcode": "synthetic-id", "qrcode_img_content": "synthetic-url"}, {"status": failure})
    monkeypatch.setattr(weixin, "WeixinClient", lambda: client)
    monkeypatch.setitem(
        sys.modules,
        "qrcode",
        SimpleNamespace(make=lambda text: SimpleNamespace(save=lambda stream, **kw: stream.write(b"fake-png"))),
    )
    with pytest.raises(TransportError):
        weixin.login(account, replace=True)
    assert account.read_bytes() == original
    assert not list(tmp_path.glob("login-*.png"))


def test_timeout_keeps_empty_checkpoint_exactly(credentials):
    """保留尚未初始化的空 checkpoint 形态。

    作用:
        超时不能伪装成首次收到有效同步位置。
    参数:
        credentials: 合成凭据。
    返回:
        None。
    行为说明:
        1. 只返回空批次，不添加新字段。
    """
    adapter = weixin.WeixinAdapter(credentials, FakeClient(weixin.WeixinTimeout("等待超时")))
    assert adapter.poll({}).checkpoint == {}
