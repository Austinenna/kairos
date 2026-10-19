"""验证微信命令只负责登录或转入统一程序，不另外创建 Runtime。"""

import json
from types import SimpleNamespace

import pytest

from kairos import program
from kairos import weixin as entry
from kairos.app_lock import AppAlreadyRunning, application_lock
from kairos.channels.weixin import login, read_credentials, write_credentials


def lock_is_held(directory):
    """用独立文件描述符确认当前运行锁。

    作用:
        验证登录和凭据导入发生在统一程序的锁内。
    参数:
        directory: 测试临时目录。
    返回:
        是否已被持锁（bool）。
    行为说明:
        1. 不读取任何真实用户状态。
    """
    try:
        with application_lock(directory):
            return False
    except AppAlreadyRunning:
        return True


@pytest.fixture
def account(isolated_app_dir):
    """在隔离目录准备仅用于离线测试的凭据。

    作用:
        允许测试真实文件导入和已有账号保护，不进行微信网络请求。
    参数:
        isolated_app_dir: 临时用户目录。
    返回:
        账号路径与合成凭据。
    行为说明:
        1. 凭据写入采用正式的私有文件权限逻辑。
    """
    path = isolated_app_dir / "channels" / "weixin" / "account.json"
    credentials = {
        "bot_token": "synthetic-bot-token",
        "bot_id": "test-bot",
        "owner_id": "test-owner",
        "base_url": "https://ilinkai.weixin.qq.com",
        "initial_checkpoint": {"cursor": "bootstrap", "context_tokens": {}},
    }
    write_credentials(path, credentials)
    return path, credentials


def test_help_and_invalid_flags_do_not_touch_data(monkeypatch, tmp_path):
    """验证帮助和非法参数先于任何用户状态访问。

    作用:
        保持无需配置即可检查命令帮助的行为。
    参数:
        monkeypatch, tmp_path: 隔离替换和路径。
    返回:
        None。
    行为说明:
        1. 数据目录和运行函数均设置为失败哨兵。
    """

    def forbidden(*args, **kwargs):
        """拒绝测试边界以外的状态访问。

        作用:
            在帮助路径发生副作用时使测试失败。
        参数:
            args, kwargs: 任意调用参数。
        返回:
            不正常返回。
        行为说明:
            1. 失败由 pytest.fail 产生。
        """
        pytest.fail("不应访问配置、凭据或启动程序")

    monkeypatch.setattr(entry.config, "get_app_dir", forbidden)
    monkeypatch.setattr(program, "run", forbidden)
    monkeypatch.setattr(entry, "login", forbidden)
    initial = set(tmp_path.rglob("*"))
    for arguments, code in [(["--help"], 0), (["run", "--replace-login"], 2), (["run", "--max-messages", "-1"], 2)]:
        with pytest.raises(SystemExit) as stopped:
            entry.main(arguments)
        assert stopped.value.code == code
    assert set(tmp_path.rglob("*")) == initial


def test_legacy_run_delegates_to_same_program(monkeypatch):
    """验证旧命令使用统一程序的仅微信模式。

    作用:
        防止保留另一套创建 Runtime 的入口。
    参数:
        monkeypatch: 捕获程序调用。
    返回:
        None。
    行为说明:
        1. 原样传递验收上限和返回状态。
    """
    calls = []
    monkeypatch.setattr(program, "run", lambda **kwargs: calls.append(kwargs) or 1)
    assert entry.main(["run", "--max-messages", "2", "--idle-timeout", "3"]) == 1
    assert calls == [{"channels": ("weixin",), "max_messages": 2, "idle_timeout": 3.0}]


def test_default_login_preserves_existing_account_under_lock(account, monkeypatch, isolated_app_dir):
    """验证默认登录拒绝覆盖已有凭据，且不初始化核心。

    作用:
        保留已有的显式替换授权边界。
    参数:
        account, monkeypatch, isolated_app_dir: 合成账号和隔离环境。
    返回:
        None。
    行为说明:
        1. 实际 login 会在访问网络前拒绝覆盖。
    """
    path, _ = account
    original = path.read_bytes()
    observed = []

    def guarded_login(target, *, replace):
        """核对持锁后调用正式登录的覆盖保护。

        作用:
            验证真实保护规则而不是模拟其结果。
        参数:
            target, replace: 账号路径和替换授权。
        返回:
            正式登录的返回值。
        行为说明:
            1. 已有账号路径在联网前拒绝。
        """
        observed.append((replace, lock_is_held(isolated_app_dir)))
        return login(target, replace=replace)

    monkeypatch.setattr(entry, "login", guarded_login)
    monkeypatch.setattr(program, "run", lambda **kwargs: pytest.fail("登录不能启动核心"))
    assert entry.main(["login"]) == 1
    assert observed == [(False, True)]
    assert path.read_bytes() == original


def test_explicit_replace_only_runs_login(account, monkeypatch, isolated_app_dir):
    """验证显式替换只执行登录，不能同时开始聊天。

    作用:
        限定替换登录的影响范围。
    参数:
        account, monkeypatch, isolated_app_dir: 合成账号和隔离替换。
    返回:
        None。
    行为说明:
        1. 不产生二维码或网络调用。
    """
    path, _ = account
    observed = []
    monkeypatch.setattr(
        entry, "login", lambda target, *, replace: observed.append((target, replace, lock_is_held(isolated_app_dir)))
    )
    monkeypatch.setattr(program, "run", lambda **kwargs: pytest.fail("登录不能启动核心"))
    assert entry.main(["login", "--replace-login"]) == 0
    assert observed == [(path, True, True)]
    assert not lock_is_held(isolated_app_dir)


def test_running_program_blocks_login(account, monkeypatch, isolated_app_dir):
    """验证登录替换不能与正在运行的程序交错。

    作用:
        防止接收线程使用旧凭据时另一个进程覆盖账号。
    参数:
        account, monkeypatch, isolated_app_dir: 隔离账号和锁。
    返回:
        None。
    行为说明:
        1. 持锁时不进入登录函数。
    """
    monkeypatch.setattr(entry, "login", lambda *args, **kwargs: pytest.fail("已有程序时不能登录"))
    with application_lock(isolated_app_dir):
        assert entry.main(["login", "--replace-login"]) == 1


def test_probe_import_preserves_original_and_existing_credentials(account, monkeypatch, isolated_app_dir):
    """验证统一程序仍保留探针导入和后续凭据不覆盖规则。

    作用:
        检查改造后的装配入口使用原有私有文件、初始游标和账号。
    参数:
        account, monkeypatch, isolated_app_dir: 合成账号与隔离目录。
    返回:
        None。
    行为说明:
        1. 两次装配都不启动收发；第二次不会回退到修改后的探针凭据。
    """
    path, credentials = account
    path.unlink()
    source = isolated_app_dir / "weixin-probe" / "state.json"
    source.parent.mkdir()
    source.write_text(json.dumps({"credentials": credentials, "cursor": "probe-cursor", "replies": {}}))
    original = source.read_bytes()
    captured = []
    monkeypatch.setattr(program, "WeixinChannelRunner", lambda *args, **kwargs: captured.append((args, kwargs)))
    service = SimpleNamespace(client=lambda *args, **kwargs: None)
    with application_lock(isolated_app_dir):
        program._weixin_runner(service, isolated_app_dir, lambda message: None, max_messages=1, idle_timeout=2)
    saved = path.read_bytes()
    assert source.read_bytes() == original
    assert read_credentials(path)["bot_token"] == credentials["bot_token"]
    assert captured[0][1]["initial_checkpoint"]["cursor"] == "probe-cursor"
    source.write_text("invalid-probe-must-not-be-read")
    with application_lock(isolated_app_dir):
        program._weixin_runner(service, isolated_app_dir, lambda message: None, max_messages=0, idle_timeout=None)
    assert path.read_bytes() == saved
