"""验证统一程序的入口装配、运行锁与停止顺序，不访问真实微信或模型。"""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from kairos import program
from kairos.app_lock import AppAlreadyRunning, application_lock
from kairos.config import Settings


@pytest.fixture
def harness(monkeypatch, isolated_app_dir):
    """建立可注入故障的装配边界，仍使用真实文件锁。

    作用:
        在每个入口和资源操作时核验锁，防止后台尚未结束就放行第二个程序。
    参数:
        monkeypatch, isolated_app_dir: 隔离替换及临时用户目录。
    返回:
        测试事件与故障设置。
    行为说明:
        1. 不创建真实网络客户端；Runtime 本身另由应用层集成测试覆盖。
    """
    state = SimpleNamespace(events=[], failures={}, reason="exit", worker_error=None, clients=[], terminal_options=None)
    state.settings = SimpleNamespace(kairos_api_key="synthetic-key")

    def record(stage):
        """记录阶段，并核对阶段执行时总锁仍然有效。

        作用:
            将生命周期测试与具体资源实现解耦。
        参数:
            stage: 固定的阶段名称。
        返回:
            None。
        异常:
            BaseException: 该阶段配置了合成故障。
        行为说明:
            1. 第二个文件描述符必须无法取得同一个锁。
        """
        state.events.append(stage)
        with pytest.raises(AppAlreadyRunning):
            with application_lock(isolated_app_dir):
                pytest.fail("资源生命周期超出了运行锁范围")
        if stage in state.failures:
            raise state.failures[stage]

    class Client:
        def close(self):
            """在仍持锁时分离终端入口。

            作用:
                验证界面结束不能替代统一服务收尾。
            返回:
                None。
            行为说明:
                1. 只记录客户端关闭，不假装关闭 Runtime。
            """
            record("client-close")

    class Service:
        def __init__(self, factory, *, request_path):
            """记录唯一服务的创建位置。

            作用:
                检查统一入口使用独立的请求持久化路径。
            参数:
                factory, request_path: Runtime 工厂及请求库位置。
            返回:
                None。
            行为说明:
                1. 不调用真实模型工厂。
            """
            record("service")
            assert request_path == isolated_app_dir / "application" / "requests.sqlite3"

        def client(self, source, **kwargs):
            """记录终端会话选择，不创建第二个核心。

            作用:
                检查入口会话参数只传给对应客户端。
            参数:
                source, kwargs: 入口及会话选项。
            返回:
                合成客户端。
            行为说明:
                1. 本测试不模拟 Runtime 状态。
            """
            record("client")
            state.clients.append((source, kwargs))
            return Client()

        def stop_accepting(self):
            """在取消之前关闭新请求准入。

            作用:
                防止收发线程结束前在取消屏障之后又提交新任务。
            返回:
                None。
            行为说明:
                1. 不关闭 Runtime，后续仍需等待线程并关闭服务。
            """
            record("stop-accepting")

        def cancel_all(self):
            """记录关闭前对未完成请求的取消。

            作用:
                确认取消发生在线程等待之前。
            返回:
                None。
            行为说明:
                1. 不代表同步工具一定已停止。
            """
            record("cancel")

        def close(self):
            """记录核心最后收尾。

            作用:
                验证微信线程结束后才关闭 Runtime。
            返回:
                None。
            行为说明:
                1. 可注入关闭失败，检查锁最终释放。
            """
            record("service-close")

    class Runner:
        error = None
        running = True

        def start(self):
            """模拟微信模块启动。

            作用:
                覆盖部分启动失败时的统一清理。
            返回:
                None。
            行为说明:
                1. 不创建线程或访问网络。
            """
            record("weixin-start")
            self.error = state.worker_error

        def stop(self):
            """记录停止接收信号。

            作用:
                确认程序先停止入口再结束核心。
            返回:
                None。
            行为说明:
                1. 停止信号不等于线程已退出。
            """
            record("weixin-stop")

        def join(self, timeout=None):
            """模拟线程确实结束的确认。

            作用:
                为生命周期顺序提供可观察边界。
            参数:
                timeout: 单次有界等待时间。
            返回:
                True。
            行为说明:
                1. 不阻塞测试。
            """
            record("weixin-join")
            return True

    def make_runner(*args, **kwargs):
        """记录微信装配但不读取真实账号。

        作用:
            在隔离边界测试程序对入口初始化失败的处理。
        参数:
            args, kwargs: 实际装配参数。
        返回:
            合成微信运行器。
        行为说明:
            1. 原始凭据导入另行测试。
        """
        record("weixin-factory")
        return Runner()

    def terminal(**kwargs):
        """模拟终端完成、分离或中断。

        作用:
            核对程序传入的是客户端而不是 Runtime 工厂。
        参数:
            kwargs: 终端界面参数。
        返回:
            配置的退出原因。
        行为说明:
            1. 不读取 stdin。
        """
        record("terminal")
        state.terminal_options = kwargs
        assert isinstance(kwargs["runtime"], Client)
        return state.reason

    monkeypatch.setattr(program, "Settings", lambda: state.settings)
    monkeypatch.setattr(program, "ApplicationService", Service)
    monkeypatch.setattr(program, "_weixin_runner", make_runner)
    monkeypatch.setattr(program.cli, "run", terminal)
    return state


@pytest.mark.parametrize("channels", [("terminal",), ("weixin",), ("terminal", "weixin")])
def test_enabled_entries_share_one_service_and_close_before_unlock(harness, channels, isolated_app_dir):
    """验证所有入口组合只装配一次核心服务，并在退出后释放总锁。

    作用:
        核对统一程序的资源归属和调用方可观察行为。
    参数:
        harness: 使用真实文件锁、可注入入口故障的隔离装配环境。
        channels: 本次启用的入口组合。
        isolated_app_dir: 测试专用用户数据目录。
    返回:
        None。
    行为说明:
        1. 通过事件顺序、状态及输出断言验证行为，不调用真实模型或微信。
    """
    assert program.run(channels=channels) == 0
    assert harness.events.count("service") == harness.events.count("service-close") == 1
    assert ("terminal" in harness.events) == ("terminal" in channels)
    assert ("weixin-start" in harness.events) == ("weixin" in channels)
    if "weixin" in channels:
        assert harness.events.index("weixin-stop") < harness.events.index("cancel")
        assert harness.events[-1] == "service-close"
    assert harness.events.index("stop-accepting") < harness.events.index("cancel")
    with application_lock(isolated_app_dir):
        pass


@pytest.mark.parametrize("legacy_selection", [None, '["terminal"]'])
def test_no_channel_argument_starts_both_even_with_legacy_setting(
    harness, isolated_app_dir, monkeypatch, legacy_selection
):
    """验证无入口参数始终启动两端，旧私有配置不能改变默认行为。

    作用:
        覆盖直接运行命令以及用户保留旧 KAIROS_CHANNELS 配置的升级场景。
    参数:
        harness: 保留真实程序装配和运行锁的离线依赖。
        isolated_app_dir: 隔离配置目录。
        monkeypatch: 恢复实际设置读取并隔离环境变量。
        legacy_selection: 可选的旧入口配置值。
    返回:
        None。
    行为说明:
        1. 从真实临时 .env 读取模型配置，只用替身阻止网络与界面交互。
        2. 检查终端和微信均实际进入启动流程，仍只创建一份应用服务。
    """
    for name in (*Settings.model_fields, "kairos_channels"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    contents = "KAIROS_API_KEY=synthetic-key\nGATEWAY_BASE_URL=https://offline.invalid/v1\nKAIROS_MODEL=offline\n"
    if legacy_selection is not None:
        contents += f"KAIROS_CHANNELS={legacy_selection}\n"
    (isolated_app_dir / ".env").write_text(contents, encoding="utf-8")
    monkeypatch.setattr(program, "Settings", Settings)

    assert program.run() == 0
    assert "terminal" in harness.events
    assert "weixin-start" in harness.events
    assert harness.events.count("service") == 1


def test_terminal_session_choice_and_detach_do_not_stop_weixin_early(harness):
    """验证终端恢复参数只归终端，detach 后先等待微信再统一收尾。

    作用:
        核对统一程序的资源归属和调用方可观察行为。
    参数:
        harness: 使用真实文件锁、可注入入口故障的隔离装配环境。
    返回:
        None。
    行为说明:
        1. 通过事件顺序、状态及输出断言验证行为，不调用真实模型或微信。
    """
    harness.reason = "detach"
    assert program.run(session_id="selected-session") == 0
    assert harness.clients == [("terminal", {"session_id": "selected-session", "resume_latest": False})]
    assert harness.terminal_options["allow_detach"] is True
    assert harness.events.index("weixin-join") < harness.events.index("weixin-stop")


@pytest.mark.parametrize("stage", ["weixin-factory", "weixin-start"])
def test_weixin_start_failure_keeps_terminal_and_redacts_error(harness, stage, capsys):
    """验证微信启动失败不会阻止终端，且原始异常不进入输出。

    作用:
        核对统一程序的资源归属和调用方可观察行为。
    参数:
        harness: 使用真实文件锁、可注入入口故障的隔离装配环境。
        stage: 注入合成错误的生命周期位置。
        capsys: 捕获输出以检查错误脱敏。
    返回:
        None。
    行为说明:
        1. 通过事件顺序、状态及输出断言验证行为，不调用真实模型或微信。
    """
    harness.failures[stage] = RuntimeError("synthetic-private-request")
    assert program.run() == 1
    assert "terminal" in harness.events
    assert harness.events[-1] == "service-close"
    assert "synthetic-private-request" not in capsys.readouterr().out


@pytest.mark.parametrize("error", [KeyboardInterrupt(), program.ProgramStopped()])
def test_interruption_waits_for_entries_before_core_close(harness, error, isolated_app_dir):
    """验证 Ctrl+C 和系统停止都走持锁收尾，不遗留工作线程所有权。

    作用:
        核对统一程序的资源归属和调用方可观察行为。
    参数:
        harness: 使用真实文件锁、可注入入口故障的隔离装配环境。
        error: 待模拟的中断或关闭异常。
        isolated_app_dir: 测试专用用户数据目录。
    返回:
        None。
    行为说明:
        1. 通过事件顺序、状态及输出断言验证行为，不调用真实模型或微信。
    """
    harness.failures["terminal"] = error
    assert program.run() == 0
    assert harness.events[-1] == "service-close"
    assert harness.events.index("weixin-join") < harness.events.index("service-close")
    with application_lock(isolated_app_dir):
        pass


@pytest.mark.parametrize("stage", ["client", "client-close", "service-close"])
def test_core_failures_are_private_and_release_lock(harness, stage, capsys, isolated_app_dir):
    """验证初始化和清理异常不回显私人内容，入口异常仍尝试关闭核心。

    作用:
        核对统一程序的资源归属和调用方可观察行为。
    参数:
        harness: 使用真实文件锁、可注入入口故障的隔离装配环境。
        stage: 注入合成错误的生命周期位置。
        capsys: 捕获输出以检查错误脱敏。
        isolated_app_dir: 测试专用用户数据目录。
    返回:
        None。
    行为说明:
        1. 通过事件顺序、状态及输出断言验证行为，不调用真实模型或微信。
    """
    harness.failures[stage] = RuntimeError("synthetic-private-core")
    assert program.run(channels=("terminal",)) == 1
    assert harness.events.count("service-close") == 1
    assert "synthetic-private-core" not in capsys.readouterr().out
    with application_lock(isolated_app_dir):
        pass


def test_existing_process_prevents_any_resource_initialization(harness, isolated_app_dir):
    """验证总锁阻止第二个程序，双入口应在同一个程序中启用。

    作用:
        核对统一程序的资源归属和调用方可观察行为。
    参数:
        harness: 使用真实文件锁、可注入入口故障的隔离装配环境。
        isolated_app_dir: 测试专用用户数据目录。
    返回:
        None。
    行为说明:
        1. 通过事件顺序、状态及输出断言验证行为，不调用真实模型或微信。
    """
    with application_lock(isolated_app_dir):
        assert program.run() == 1
    assert harness.events == []


def test_invalid_settings_never_echo_secret(monkeypatch, capsys):
    """验证配置校验失败不回显 Pydantic 原始输入。

    作用:
        核对统一程序的资源归属和调用方可观察行为。
    参数:
        monkeypatch: 测试结束后自动恢复的依赖替换。
        capsys: 捕获输出以检查错误脱敏。
    返回:
        None。
    行为说明:
        1. 通过事件顺序、状态及输出断言验证行为，不调用真实模型或微信。
    """
    error = ValidationError.from_exception_data(
        "Settings", [{"type": "int_parsing", "loc": ("anthropic_max_tokens",), "input": "synthetic-secret"}]
    )

    def broken_settings():
        """用带合成秘密的配置异常检查输出边界。

        作用:
            为实际程序入口提供不访问网络的隔离替身。
        返回:
            None。
        异常:
            ValidationError: 无条件抛出带合成输入的配置异常。
        行为说明:
            1. 资源管理仍由被测程序执行，替身不操作个人数据。
        """
        raise error

    monkeypatch.setattr(program, "Settings", broken_settings)
    assert program.run() == 1
    output = capsys.readouterr().out
    assert "配置无效" in output
    assert "synthetic-secret" not in output


@pytest.mark.parametrize("channels", ["", "terminal,", "unknown", []])
def test_invalid_channel_selection_rejected(channels):
    """拒绝空入口和未知入口，不静默退回某个运行方式。

    作用:
        核对统一程序的资源归属和调用方可观察行为。
    参数:
        channels: 本次启用的入口组合。
    返回:
        None。
    行为说明:
        1. 通过事件顺序、状态及输出断言验证行为，不调用真实模型或微信。
    """
    with pytest.raises(ValueError):
        program.parse_channels(channels)
