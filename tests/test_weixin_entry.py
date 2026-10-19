"""验证微信命令入口的真实文件锁、初始化与安全收尾边界。"""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from kairos import weixin as entry
from kairos.app_lock import AppAlreadyRunning, application_lock
from kairos.channels.base import ReceiveBatch, TransportError
from kairos.channels.store import ChannelStore
from kairos.channels.weixin import import_probe, login, read_credentials, write_credentials


def lock_is_held(directory):
    """用第二个独立文件描述符检查数据目录锁。

    作用:
        通过真实 flock 验证入口生命周期，不依赖入口内部实现状态。
    参数:
        directory: 测试专用临时数据目录。
    返回:
        是否已有另一处持有排他锁 (bool)。
    行为说明:
        1. 未持锁时短暂取得并释放测试锁，不访问真实用户目录。
    """
    try:
        with application_lock(directory, label="离线锁检查"):
            return False
    except AppAlreadyRunning:
        return True


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """构造假模型和假协调器，保留真实通道存储及文件锁。

    作用:
        验证入口是否在正确锁范围初始化、执行及关闭资源。
    参数:
        tmp_path: 唯一允许读写的合成用户目录。
        monkeypatch: 替换配置、适配器、协调器和运行时工厂。
    返回:
        持有事件、注入错误与测试路径的简单对象。
    行为说明:
        1. 不读取 .env，不创建真实模型客户端，不访问网络。
        2. 凭据和 SQLite 都使用隔离临时文件。
    """
    state = SimpleNamespace(
        directory=tmp_path / "isolated-kairos",
        events=[],
        failures={},
        next_checkpoint=None,
        runtime_arguments=[],
        run_arguments=[],
        observed_checkpoints=[],
    )
    state.account = state.directory / "channels" / "weixin" / "account.json"
    state.credentials = {
        "bot_token": "synthetic-bot-token",
        "bot_id": "test-bot",
        "owner_id": "test-owner",
        "base_url": "https://ilinkai.weixin.qq.com",
        "initial_checkpoint": {"cursor": "bootstrap", "context_tokens": {}},
    }
    state.settings = SimpleNamespace(kairos_api_key="synthetic-model-key")
    write_credentials(state.account, state.credentials)

    def record(stage):
        """记录阶段是否持锁，并在指定阶段注入失败。

        作用:
            为正常、初始化失败和关闭失败路径提供统一观测。
        参数:
            stage: 固定阶段名称。
        返回:
            None。
        异常:
            BaseException: 当前阶段配置了待注入的异常。
        行为说明:
            1. 每次记录都使用第二个描述符验证真实锁。
        """
        state.events.append((stage, lock_is_held(state.directory)))
        if stage in state.failures:
            raise state.failures[stage]

    def settings_factory():
        """返回合成配置并检查读取时机。

        作用:
            禁止测试实例化实际用户设置。
        参数:
            无。
        返回:
            合成设置对象。
        行为说明:
            1. 配置失败由 record 注入。
        """
        record("settings")
        return state.settings

    def read_account(path):
        """在记录锁状态后读取合成凭据。

        作用:
            检查正式账号文件读取发生在锁内。
        参数:
            path: 入口选择的账号路径。
        返回:
            隔离文件中的凭据。
        行为说明:
            1. 仅允许读取 fixture 创建的路径。
        """
        record("read-account")
        assert path == state.account
        return read_credentials(path)

    def copy_probe(source, target):
        """在真实探针导入前记录锁状态。

        作用:
            检查首次复制凭据不能越过应用互斥边界。
        参数:
            source: 隔离探针文件路径。
            target: 正式账号文件路径。
        返回:
            导入后可恢复的初始 checkpoint。
        行为说明:
            1. 仅使用测试显式创建的探针文件。
        """
        record("import-probe")
        assert source == state.directory / "weixin-probe" / "state.json"
        assert target == state.account
        return import_probe(source, target)

    def adapter_factory(credentials):
        """返回无网络功能的通道身份对象。

        作用:
            防止测试意外建立微信 HTTPS 请求。
        参数:
            credentials: 隔离凭据。
        返回:
            仅含路由身份的对象。
        行为说明:
            1. 适配器初始化错误由 record 注入。
        """
        record("adapter")
        return SimpleNamespace(channel="weixin", account_id=credentials["bot_id"], owner_id=credentials["owner_id"])

    class TrackingStore(ChannelStore):
        """在真实 SQLite 资源生命周期上记录应用锁。"""

        def __init__(self, path):
            """在锁状态观测后打开隔离数据库。

            作用:
                检查入口创建数据库的顺序。
            参数:
                path: 测试专用通道 SQLite 路径。
            返回:
                无返回值。
            行为说明:
                1. 数据库语义使用真实 ChannelStore。
            """
            record("store")
            assert path == state.directory / "channels" / "channels.sqlite3"
            super().__init__(path)

        def seed_checkpoint(self, channel, account, checkpoint):
            """记录并执行真实初始游标写入。

            作用:
                验证重启后不会用 bootstrap 覆盖已推进的游标。
            参数:
                channel, account, checkpoint: 入口提供的初始状态。
            返回:
                None。
            行为说明:
                1. 使用真实插入逻辑，避免假存储掩盖覆盖错误。
            """
            record("seed")
            super().seed_checkpoint(channel, account, checkpoint)

        def close(self):
            """关闭真实连接并记录锁是否仍然持有。

            作用:
                检查成功和失败路径只关闭一次且先关闭后解锁。
            参数:
                无。
            返回:
                None。
            行为说明:
                1. 先释放连接，再允许注入关闭异常，避免测试泄漏资源。
            """
            super().close()
            record("store-close")

    class FakeRuntime:
        """只记录关闭事件，不运行模型。"""

        def close(self):
            """记录运行时关闭阶段。

            作用:
                验证模型资源的收尾处在应用锁内。
            参数:
                无。
            返回:
                None。
            行为说明:
                1. 不连接网络或业务数据库。
            """
            record("runtime-close")

    def runtime_factory(**kwargs):
        """检查入口注入的路径并返回假运行时。

        作用:
            验证初始化失败不会输出模型凭据或私人路径。
        参数:
            kwargs: settings、session_id 与隔离路径。
        返回:
            FakeRuntime。
        行为说明:
            1. 仅记录参数，绝不创建模型客户端。
        """
        record("runtime")
        assert kwargs["settings"] is state.settings
        assert kwargs["db_path"] == state.directory / "kairos.sqlite3"
        assert kwargs["memory_dir"] == state.directory / "memory"
        state.runtime_arguments.append(kwargs)
        return FakeRuntime()

    class FakeApp:
        """驱动入口生命周期，不执行真实通道收发。"""

        def __init__(self, store, adapter, factory, *, report):
            """绑定假运行时工厂与真实隔离存储。

            作用:
                检查协调器构造异常后的数据库清理。
            参数:
                store, adapter, factory, report: 入口传入的依赖。
            返回:
                无返回值。
            行为说明:
                1. 构造阶段不启动模型。
            """
            record("app")
            self.store = store
            self.adapter = adapter
            self.factory = factory
            self.runtime = None

        def run(self, *, max_messages, idle_timeout):
            """记录当前游标并模拟一次运行。

            作用:
                覆盖入口参数传递、初始游标及重启恢复边界。
            参数:
                max_messages: 验收输入上限。
                idle_timeout: 空闲上限。
            返回:
                处理数 1。
            行为说明:
                1. 可推进真实 checkpoint，但不读取任何微信消息。
            """
            record("run")
            state.run_arguments.append((max_messages, idle_timeout))
            state.observed_checkpoints.append(self.store.checkpoint("weixin", "test-bot"))
            self.runtime = self.factory(session_id="synthetic-session")
            if state.next_checkpoint is not None:
                self.store.accept_batch("weixin", "test-bot", ReceiveBatch((), state.next_checkpoint))
            return 1

        def close(self):
            """关闭假运行时并记录协调器关闭阶段。

            作用:
                检查资源顺序以及协调器关闭失败时的数据库收尾。
            参数:
                无。
            返回:
                None。
            行为说明:
                1. 不关闭通道数据库，它由入口负责。
            """
            if self.runtime is not None:
                self.runtime.close()
            record("app-close")

    monkeypatch.setattr(entry.config, "get_app_dir", lambda: state.directory)
    monkeypatch.setattr(entry, "Settings", settings_factory)
    monkeypatch.setattr(entry, "read_credentials", read_account)
    monkeypatch.setattr(entry, "import_probe", copy_probe)
    monkeypatch.setattr(entry, "WeixinAdapter", adapter_factory)
    monkeypatch.setattr(entry, "ChannelStore", TrackingStore)
    monkeypatch.setattr(entry, "ChannelApp", FakeApp)
    monkeypatch.setattr(entry, "create_runtime", runtime_factory)
    return state


def test_help_and_invalid_replace_flag_do_not_touch_configuration_or_disk(tmp_path, monkeypatch, capsys):
    """在帮助或非法替换命令时停止于任何状态读取之前。

    作用:
        保证查询帮助不会触发登录、模型配置或目录创建。
    参数:
        tmp_path: 用于确认没有新文件的隔离目录。
        monkeypatch: 禁止配置与资源入口被调用。
        capsys: 捕获 argparse 输出。
    返回:
        None。
    行为说明:
        1. 同时检查 --replace-login 在 run 下不能发生副作用。
    """

    def forbidden(*args, **kwargs):
        """拒绝不应发生的状态访问。

        作用:
            为无需初始化的命令提供失败哨兵。
        参数:
            args, kwargs: 任意调用参数。
        返回:
            不返回。
        行为说明:
            1. 调用即令测试失败。
        """
        pytest.fail("帮助和参数拒绝路径不能访问用户数据或配置")

    monkeypatch.setattr(entry.config, "get_app_dir", forbidden)
    for name in ("Settings", "application_lock", "login", "import_probe", "read_credentials", "create_runtime"):
        monkeypatch.setattr(entry, name, forbidden)
    initial_paths = set(tmp_path.rglob("*"))
    with pytest.raises(SystemExit) as help_exit:
        entry.main(["--help"])
    assert help_exit.value.code == 0
    with pytest.raises(SystemExit) as invalid_exit:
        entry.main(["run", "--replace-login"])
    assert invalid_exit.value.code == 2
    assert "仅用于 login" in capsys.readouterr().err
    assert set(tmp_path.rglob("*")) == initial_paths


def test_run_closes_all_resources_under_lock_without_overwriting_account(harness):
    """验证已登录入口的完整持锁生命周期。

    作用:
        防止覆盖凭据、重复导入和资源未关闭即允许另一入口写数据。
    参数:
        harness: 隔离入口依赖。
    返回:
        None。
    行为说明:
        1. 从配置读取到最后关闭都用真实 flock 验证。
    """
    original = harness.account.read_bytes()
    assert entry.main(["run", "--max-messages", "1", "--idle-timeout", "2"]) == 0
    assert harness.account.read_bytes() == original
    assert "import-probe" not in [stage for stage, _ in harness.events]
    assert all(held for _, held in harness.events)
    assert harness.events[-3:] == [("runtime-close", True), ("app-close", True), ("store-close", True)]
    assert harness.run_arguments == [(1, 2.0)]
    assert not lock_is_held(harness.directory)


def test_existing_cli_lock_prevents_import_and_runtime_initialization(harness):
    """已有 CLI 持锁时拒绝任何微信初始化。

    作用:
        从命令入口验证同一用户数据库的互斥。
    参数:
        harness: 隔离入口依赖。
    返回:
        None。
    行为说明:
        1. 即使缺少账号，也不会导入探针或读取模型配置。
    """
    harness.account.unlink()
    with application_lock(harness.directory, label="合成 CLI"):
        assert entry.main(["run"]) == 1
        assert harness.events == []
        assert not harness.account.exists()


def test_probe_copy_and_bootstrap_happen_under_lock_and_do_not_rewind_later_cursor(harness):
    """验证首次探针迁移及再次启动的游标边界。

    作用:
        保证已处理的微信输入不会因初始 checkpoint 被再次送给模型。
    参数:
        harness: 隔离入口依赖。
    返回:
        None。
    行为说明:
        1. 使用真实导入与真实 SQLite，第二次运行必须读到已推进游标。
    """
    harness.account.unlink()
    source = harness.directory / "weixin-probe" / "state.json"
    source.parent.mkdir()
    source.write_text(json.dumps({"credentials": harness.credentials, "cursor": "probe-cursor", "replies": {}}))
    original = source.read_bytes()
    later = {"cursor": "later-cursor", "context_tokens": {"test-owner": "synthetic-later-context"}}
    harness.next_checkpoint = later
    assert entry.main(["run", "--max-messages", "1"]) == 0
    assert harness.observed_checkpoints == [{"cursor": "probe-cursor", "context_tokens": {}}]
    assert source.read_bytes() == original
    assert ("import-probe", True) in harness.events
    harness.next_checkpoint = None
    assert entry.main(["run", "--max-messages", "1"]) == 0
    assert harness.observed_checkpoints[-1] == later
    assert sum(stage == "import-probe" for stage, _ in harness.events) == 1
    assert all(held for _, held in harness.events)


@pytest.mark.parametrize("stage", ["settings", "read-account", "adapter", "store", "seed", "app", "runtime"])
def test_initialization_errors_are_redacted_and_close_open_store_while_locked(harness, capsys, stage):
    """在各初始化阶段注入带秘密文本的错误。

    作用:
        验证异常输出脱敏以及已创建数据库的锁内关闭。
    参数:
        harness: 隔离入口依赖。
        capsys: 捕获终端输出。
        stage: 初始化失败位置。
    返回:
        None。
    行为说明:
        1. 只有已成功创建的资源应被关闭，且不重复关闭。
    """
    secret = "synthetic-private-key-and-request-body"
    harness.failures[stage] = RuntimeError(secret)
    assert entry.main(["run"]) == 1
    output = capsys.readouterr()
    assert secret not in output.out + output.err
    assert all(held for _, held in harness.events)
    closes = [name for name, _ in harness.events if name == "store-close"]
    assert len(closes) == (1 if stage in ("seed", "app", "runtime") else 0)
    assert not lock_is_held(harness.directory)


def test_settings_validation_error_does_not_echo_invalid_secret_value(harness, capsys):
    """使用真实 Pydantic ValidationError 检查配置错误输出。

    作用:
        防止 Pydantic 默认错误文本回显包含凭据的输入值。
    参数:
        harness: 隔离入口依赖。
        capsys: 捕获终端输出。
    返回:
        None。
    行为说明:
        1. 不构造真实 Settings，不读取 .env。
    """
    secret = "synthetic-validation-secret"
    error = ValidationError.from_exception_data(
        "Settings", [{"type": "int_parsing", "loc": ("anthropic_max_tokens",), "input": secret}]
    )
    assert secret in str(error)
    harness.failures["settings"] = error
    assert entry.main(["run"]) == 1
    output = capsys.readouterr()
    assert "模型配置无效" in output.out
    assert secret not in output.out + output.err
    assert harness.events == [("settings", True)]
    assert not lock_is_held(harness.directory)


@pytest.mark.parametrize(
    "stage,error,expected",
    [
        ("run", KeyboardInterrupt(), 0),
        ("run", TransportError("受控网络故障", retryable=True), 1),
        ("app-close", RuntimeError("synthetic-close-secret"), 1),
        ("store-close", RuntimeError("synthetic-close-secret"), 1),
    ],
)
def test_run_and_close_failures_release_resources_once_before_unlock(harness, capsys, stage, error, expected):
    """验证运行中断和关闭异常的收尾顺序。

    作用:
        即使某项关闭失败，后续资源仍须在持锁期间完成收尾。
    参数:
        harness: 隔离入口依赖。
        capsys: 捕获终端输出。
        stage: 失败位置。
        error: 待注入异常。
        expected: 对外退出码。
    返回:
        None。
    行为说明:
        1. app 与 store 均至多关闭一次，最终共享锁释放。
    """
    harness.failures[stage] = error
    assert entry.main(["run"]) == expected
    assert all(held for _, held in harness.events)
    stages = [name for name, _ in harness.events]
    assert stages.count("app-close") == stages.count("store-close") == 1
    assert stages[-1] == "store-close"
    output = capsys.readouterr()
    assert "synthetic-close-secret" not in output.out + output.err
    assert not lock_is_held(harness.directory)


def test_default_login_keeps_existing_credentials_without_loading_model_settings(harness, monkeypatch):
    """默认 login 不覆盖已存在账号，也不加载模型设置。

    作用:
        验证命令入口把替换授权边界传给真实登录函数。
    参数:
        harness: 隔离入口依赖。
        monkeypatch: 在真实登录调用外记录锁状态。
    返回:
        None。
    行为说明:
        1. 真实登录函数在任何二维码或网络调用之前拒绝覆盖。
    """
    observed = []

    def existing_login(path, *, replace):
        """记录锁后调用不会联网的已有账号拒绝分支。

        作用:
            核对入口传递的替换开关。
        参数:
            path: 合成账号路径。
            replace: 入口解析的显式替换授权。
        返回:
            无正常返回。
        行为说明:
            1. 实际拒绝行为由原登录函数执行。
        """
        observed.append((replace, lock_is_held(harness.directory)))
        return login(path, replace=replace)

    monkeypatch.setattr(entry, "login", existing_login)
    original = harness.account.read_bytes()
    assert entry.main(["login"]) == 1
    assert observed == [(False, True)]
    assert harness.account.read_bytes() == original
    assert harness.events == []


def test_explicit_replace_login_runs_only_login_under_shared_lock(harness, monkeypatch):
    """将显式替换登录授权限定在登录入口。

    作用:
        确认替换登录不同时启动模型或收消息循环。
    参数:
        harness: 隔离入口依赖。
        monkeypatch: 替换真正需要用户扫码的登录动作。
    返回:
        None。
    行为说明:
        1. 只记录调用，不获取二维码、不发送消息。
    """
    observed = []

    def fake_login(path, *, replace):
        """记录替换授权和持锁状态。

        作用:
            阻止入口测试触发真实扫码。
        参数:
            path: 合成账号路径。
            replace: 显式授权开关。
        返回:
            None。
        行为说明:
            1. 不写文件。
        """
        observed.append((path, replace, lock_is_held(harness.directory)))

    monkeypatch.setattr(entry, "login", fake_login)
    assert entry.main(["login", "--replace-login"]) == 0
    assert observed == [(harness.account, True, True)]
    assert harness.events == []
    assert not lock_is_held(harness.directory)
