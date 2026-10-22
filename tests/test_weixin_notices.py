"""验证微信接收状态的去重、真实恢复和安全错误提示，不访问网络或个人数据。"""

import io
import json
from contextlib import closing
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from kairos.channels.base import TransportError
from kairos.channels.store import ChannelStore
from kairos.channels.weixin import WeixinAdapter, WeixinClient
from kairos.weixin_channel import WeixinChannelRunner

PRIVATE = "synthetic-private-value"
RECOVERED = "微信连接已恢复，继续接收消息。"


def http_error(code):
    """构造带合成敏感内容的 HTTP 失败。

    作用:
        验证通知只暴露状态码，不复制 URL、错误原文或响应正文。
    参数:
        code: 合成 HTTP 状态码。
    返回:
        HTTPError。
    行为说明:
        1. 不建立真实连接。
    """
    return HTTPError(f"https://example.invalid/{PRIVATE}", code, PRIVATE, {}, io.BytesIO(PRIVATE.encode()))


@pytest.fixture
def receive_script(tmp_path, monkeypatch):
    """提供真实接收链路的离线执行器。

    作用:
        串联客户端、适配器、接收循环与临时 SQLite，只替换网络响应和等待。
    参数:
        tmp_path: 隔离数据库目录。
        monkeypatch: 仅替换本次运行器的等待方法。
    返回:
        接受响应序列并返回运行器、通知和等待记录的函数。
    行为说明:
        1. 响应耗尽后停止并模拟超时，不伪造成功或创建模型客户端。
    """

    def run(responses):
        """执行一段合成网络响应序列。

        作用:
            观察完整错误转换及状态通知，不启动实际工作线程。
        参数:
            responses: JSON 对象、原始字节、异常或接收 runner 的回调组成的序列。
        返回:
            包含 runner、notices 和 delays 的测试状态。
        行为说明:
            1. 使用真实终止异常处理，所有数据仅写入临时目录。
        """
        events = iter(responses)
        notices, delays = [], []
        credentials = {
            "bot_token": PRIVATE,
            "bot_id": "test-bot",
            "owner_id": "test-owner",
            "base_url": "https://ilinkai.weixin.qq.com",
        }
        client = WeixinClient(credentials["base_url"], credentials["bot_token"])
        adapter = WeixinAdapter(credentials, client)
        runner = WeixinChannelRunner(
            tmp_path / "channels.sqlite3",
            lambda: adapter,
            lambda **kwargs: pytest.fail("接收状态变化不应执行模型"),
            report=notices.append,
        )

        def open_response(request, timeout):
            """返回下一条合成网络结果。

            作用:
                在 urllib 边界注入失败，让生产客户端执行脱敏和错误分类。
            参数:
                request: 客户端构造的请求。
                timeout: 请求的等待上限。
            返回:
                BytesIO 响应。
            异常:
                TimeoutError: 响应耗尽，结束此次模拟。
                Exception: 当前合成结果为异常。
            行为说明:
                1. 不发送请求；正常响应按 JSON 编码。
            """
            try:
                response = next(events)
            except StopIteration:
                runner.stop()
                raise TimeoutError(PRIVATE) from None
            if callable(response):
                response = response(runner)
            if isinstance(response, Exception):
                raise response
            return io.BytesIO(response if isinstance(response, bytes) else json.dumps(response).encode())

        client.opener = SimpleNamespace(open=open_response)
        monkeypatch.setattr(runner.stop_event, "wait", delays.append)
        runner._worker(runner._receive)
        assert PRIVATE not in "\n".join(notices)
        return SimpleNamespace(runner=runner, notices=notices, delays=delays)

    return run


def test_consecutive_failures_report_first_reason_and_final_stop(receive_script):
    """合并连续重试，同时保留首个原因和最终停止原因。

    作用:
        防止相同故障刷屏，也不改变既有退避和五次失败上限。
    参数:
        receive_script: 离线接收执行器。
    返回:
        None。
    行为说明:
        1. 最后一次改为 429，验证停止提示使用最终错误。
    """
    result = receive_script([http_error(503) for _ in range(4)] + [http_error(429)])
    assert result.notices == [
        "微信接收暂时不可用：微信 HTTP 请求失败，状态码 503；正在等待重连，已保存输入不会重跑。",
        "微信通道已停止：微信 HTTP 请求失败，状态码 429",
    ]
    assert result.delays == [2, 4, 8, 16]
    assert str(result.runner.error) == "微信 HTTP 请求失败，状态码 429"
    assert result.runner.stop_event.is_set()


def test_real_empty_response_recovers_once_and_new_outage_reports_again(receive_script):
    """仅在真实响应后通知恢复，并允许下一次故障重新提示。

    作用:
        区分一个故障内的重复重试与恢复后的新故障。
    参数:
        receive_script: 离线接收执行器。
    返回:
        None。
    行为说明:
        1. 成功批次可以没有消息，多次成功不会重复报告恢复。
    """
    result = receive_script([http_error(503), http_error(503), {}, {}, URLError(PRIVATE), {}])
    assert result.notices == [
        "微信接收暂时不可用：微信 HTTP 请求失败，状态码 503；正在等待重连，已保存输入不会重跑。",
        RECOVERED,
        "微信接收暂时不可用：微信网络请求失败；正在等待重连，已保存输入不会重跑。",
        RECOVERED,
    ]
    assert [delay for delay in result.delays if delay > 0.2] == [2, 4, 2]
    assert result.runner.error is None


def test_failures_alternating_with_timeouts_do_not_fake_recovery(receive_script):
    """合并错误与超时交替期间的重复通知。

    作用:
        回归截图中的八条重复提示，防止超时被误判为已恢复连接。
    参数:
        receive_script: 离线接收执行器。
    返回:
        None。
    行为说明:
        1. 超时仍沿用原有失败计数规则，不额外触发停止或恢复。
    """
    result = receive_script([event for _ in range(8) for event in (http_error(503), TimeoutError(PRIVATE))])
    assert result.notices == ["微信接收暂时不可用：微信 HTTP 请求失败，状态码 503；正在等待重连，已保存输入不会重跑。"]
    assert [delay for delay in result.delays if delay > 0.2] == [2] * 8
    assert result.runner.error is None


def test_normal_polling_and_timeouts_stay_quiet(receive_script):
    """保持正常空轮询和等待超时静默。

    作用:
        防止没有故障时也持续通知连接恢复。
    参数:
        receive_script: 离线接收执行器。
    返回:
        None。
    行为说明:
        1. 同时覆盖直接超时与 urllib 包装的超时。
    """
    result = receive_script([{}, TimeoutError(PRIVATE), URLError(TimeoutError(PRIVATE)), {}])
    assert result.notices == []
    assert result.runner.error is None


@pytest.mark.parametrize(
    "response,reason",
    [
        (http_error(503), "微信 HTTP 请求失败，状态码 503"),
        (URLError(PRIVATE), "微信网络请求失败"),
        ({"ret": -2, "errmsg": PRIVATE}, "微信返回业务错误：ret=-2"),
    ],
)
def test_retry_notice_shows_only_safe_error_reason(receive_script, response, reason):
    """显示安全错误类别或状态码，不输出上游敏感原文。

    作用:
        覆盖 HTTP、网络及业务失败经过生产脱敏边界后的展示。
    参数:
        receive_script: 离线接收执行器。
        response: 包含合成敏感值的上游错误。
        reason: 允许展示的受控原因。
    返回:
        None。
    行为说明:
        1. 测试执行器同时断言凭据和错误原文未进入通知。
    """
    result = receive_script([response])
    assert result.notices == [f"微信接收暂时不可用：{reason}；正在等待重连，已保存输入不会重跑。"]
    assert result.runner.error is None


@pytest.mark.parametrize(
    "response,reason,authentication",
    [
        (http_error(401), "微信 HTTP 请求失败，状态码 401", True),
        ({"ret": -14, "errmsg": PRIVATE}, "微信登录凭据已失效，请重新登录", True),
        (PRIVATE.encode(), "微信响应不是有效 JSON", False),
        (ValueError(PRIVATE), "ValueError；未输出原始错误内容", False),
    ],
)
def test_terminal_error_reports_stop_without_retry(receive_script, response, reason, authentication):
    """明确报告不可恢复错误导致的停止，不进入重连循环。

    作用:
        保持认证、格式和未知异常边界，并避免原始异常泄露。
    参数:
        receive_script: 离线接收执行器。
        response: 导致立即停止的上游错误。
        reason: 预期安全原因。
        authentication: 是否为认证失败。
    返回:
        None。
    行为说明:
        1. 不等待、不重试，最终错误仍保留给程序入口。
    """
    result = receive_script([response])
    assert result.notices == [f"微信通道已停止：{reason}"]
    assert result.delays == []
    assert str(result.runner.error) == reason
    assert result.runner.error.authentication is authentication


def test_second_worker_error_does_not_repeat_stop_notice(receive_script):
    """只报告收发工作线程的首个终止错误。

    作用:
        防止通道停止时两个工作线程各输出一条停止提示。
    参数:
        receive_script: 离线接收执行器。
    返回:
        None。
    行为说明:
        1. 后续错误不覆盖首个错误，也不输出原文。
    """
    result = receive_script([http_error(401)])
    first_error = result.runner.error

    def late_worker():
        """模拟另一个工作线程稍后失败。

        作用:
            验证已进入停止状态后的通知去重。
        返回:
            不正常返回。
        异常:
            RuntimeError: 始终抛出合成失败。
        行为说明:
            1. 不执行任务或网络请求。
        """
        raise RuntimeError(PRIVATE)

    result.runner._worker(late_worker)
    assert result.runner.error is first_error
    assert result.notices == ["微信通道已停止：微信 HTTP 请求失败，状态码 401"]


def stop_during_poll(runner, worker_failure):
    """在一个尚未返回的轮询中请求通道停止。

    作用:
        模拟用户停止，或另一工作线程失败时接收请求仍未完成的窗口。
    参数:
        runner: 当前接收运行器。
        worker_failure: 为真时通过工作线程错误处理请求停止。
    返回:
        None。
    行为说明:
        1. 不强制中断当前轮询，让测试继续返回迟到结果。
    """
    if not worker_failure:
        runner.stop()
        return

    def failed_worker():
        """模拟处理线程发生不可恢复的安全错误。

        作用:
            经真实工作线程错误处理产生最终停止通知。
        返回:
            不正常返回。
        异常:
            TransportError: 始终抛出合成安全错误。
        行为说明:
            1. 不执行实际任务。
        """
        raise TransportError("合成处理线程失败")

    runner._worker(failed_worker)


@pytest.mark.parametrize("worker_failure", [False, True])
def test_late_success_after_stop_is_saved_without_recovery_notice(receive_script, worker_failure):
    """停止后保存迟到的完整批次，但不再宣称连接恢复。

    作用:
        防止最终停止提示后追加恢复通知，同时保留已收到的输入和游标。
    参数:
        receive_script: 离线接收执行器。
        worker_failure: 区分用户停止与另一工作线程失败。
    返回:
        None。
    行为说明:
        1. 首次失败进入重连状态，再在下一次轮询返回前触发停止。
    """

    def late_response(runner):
        """在停止后返回一条有效本人消息。

        作用:
            模拟已发出的网络请求在通道停止后才返回。
        参数:
            runner: 当前运行器。
        返回:
            包含本人消息及新游标的合成响应。
        行为说明:
            1. 消息仍应原子落盘，但不再发布恢复通知。
        """
        stop_during_poll(runner, worker_failure)
        return {
            "get_updates_buf": "late-cursor",
            "msgs": [
                {
                    "message_id": "late-message",
                    "from_user_id": "test-owner",
                    "to_user_id": "test-bot",
                    "message_type": 1,
                    "context_token": "late-context",
                    "item_list": [{"type": 1, "text_item": {"text": "迟到的输入"}}],
                }
            ],
        }

    result = receive_script([http_error(503), late_response])
    expected = ["微信接收暂时不可用：微信 HTTP 请求失败，状态码 503；正在等待重连，已保存输入不会重跑。"]
    if worker_failure:
        expected.append("微信通道已停止：合成处理线程失败")
    assert result.notices == expected
    with closing(ChannelStore(result.runner.store_path)) as store:
        record = store.next_queued()
        assert record.message.message_id == "late-message"
        assert record.message.text == "迟到的输入"
        assert record.status == "queued"
        assert store.checkpoint("weixin", "test-bot") == {
            "cursor": "late-cursor",
            "context_tokens": {"test-owner": "late-context"},
        }


@pytest.mark.parametrize("worker_failure", [False, True])
def test_late_retryable_error_after_stop_does_not_report_or_wait(receive_script, worker_failure):
    """停止后忽略迟到的可重试错误，不追加重连提示或等待。

    作用:
        防止最终停止通知被随后的接收失败误导为仍在重连。
    参数:
        receive_script: 离线接收执行器。
        worker_failure: 区分用户停止与另一工作线程失败。
    返回:
        None。
    行为说明:
        1. 第一次轮询返回前停止，确保迟到错误不会产生首次重连通知。
    """

    def late_failure(runner):
        """停止通道后返回可重试网络错误。

        作用:
            在实际 urllib 错误转换之前触发停止状态。
        参数:
            runner: 当前运行器。
        返回:
            HTTP 503 异常，由测试网络边界抛出。
        行为说明:
            1. 不替换生产接收循环的错误分支。
        """
        stop_during_poll(runner, worker_failure)
        return http_error(503)

    result = receive_script([late_failure])
    assert result.notices == (["微信通道已停止：合成处理线程失败"] if worker_failure else [])
    assert result.delays == []
    assert (str(result.runner.error) if result.runner.error else None) == (
        "合成处理线程失败" if worker_failure else None
    )
