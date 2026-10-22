"""用临时目录验证入口互斥，不读取真实用户数据或启动模型。"""

import json
import os
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from kairos import cli, config, program
from kairos.app_lock import AppAlreadyRunning, application_lock
from kairos.program import legacy_main as main

CHILD_SCRIPT = """
import os
import sys
from pathlib import Path
from kairos.app_lock import AppAlreadyRunning, application_lock

try:
    with application_lock(Path(sys.argv[1]), label="Kairos test child"):
        if sys.argv[2] == "crash":
            os._exit(7)
except AppAlreadyRunning:
    sys.exit(3)
"""


def _run_child(data_dir, *, crash=False):
    """启动一个只尝试取得运行锁的独立进程。

    作用:
        检查操作系统跨进程锁语义，子进程仅访问测试临时目录。

    参数:
        data_dir: 本测试的隔离数据目录。
        crash: 为 True 时在持锁期间直接退出，模拟来不及清理的进程终止。

    返回:
        subprocess.CompletedProcess，退出码 0 表示成功，3 表示锁被占用，7 表示异常退出。

    行为说明:
        1. 使用当前 Python 解释器加载同一锁实现，限制最长运行时间为 10 秒。
    """
    return subprocess.run(
        [sys.executable, "-c", CHILD_SCRIPT, str(data_dir), "crash" if crash else "normal"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_new_directories_and_lock_file_are_private(tmp_path):
    """验证新建的运行状态只允许当前用户访问。

    作用:
        检查数据目录创建及锁内记录不携带路径或其他个人数据。

    参数:
        tmp_path: 隔离目录。

    返回:
        None。

    行为说明:
        1. 在两层缺失目录中加锁，检查目录 0700、文件 0600 和 PID、固定名称。
    """
    data_dir = tmp_path / "new-parent" / "app"
    with application_lock(data_dir, label="Kairos test"):
        assert stat.S_IMODE(data_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(data_dir.parent.stat().st_mode) == 0o700
        lock_path = data_dir / ".runtime.lock"
        assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600
        assert json.loads(lock_path.read_text()) == {"pid": os.getpid(), "label": "Kairos test"}


def test_existing_directory_permissions_are_preserved(tmp_path):
    """验证运行锁不改写已有数据目录的权限。

    作用:
        保持调用方对已有目录的访问设置。

    参数:
        tmp_path: 隔离目录。

    返回:
        None。

    行为说明:
        1. 为临时目录设置 0750，再次加锁后权限保持不变。
    """
    tmp_path.chmod(0o750)
    with application_lock(tmp_path):
        assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o750


def test_independent_process_cannot_enter_until_owner_releases(tmp_path):
    """验证两个进程不能同时进入同一数据目录。

    作用:
        覆盖 CLI 与微信服务分处不同进程时的互斥约束。

    参数:
        tmp_path: 隔离目录。

    返回:
        None。

    行为说明:
        1. 父进程持锁时，子进程立即拒绝且不覆盖原持锁者记录。
        2. 父进程释放后，新的子进程可以取得同一文件锁。
    """
    with application_lock(tmp_path, label="Kairos parent"):
        original = (tmp_path / ".runtime.lock").read_bytes()
        result = _run_child(tmp_path)
        assert result.returncode == 3, result.stderr
        assert (tmp_path / ".runtime.lock").read_bytes() == original
    result = _run_child(tmp_path)
    assert result.returncode == 0, result.stderr


def test_runtime_exception_releases_lock(tmp_path):
    """验证入口内部抛出异常后会释放运行锁。

    作用:
        避免一次失败阻止后续启动。

    参数:
        tmp_path: 隔离目录。

    返回:
        None。

    异常:
        ValueError: 测试主动模拟入口执行失败，由 pytest 捕获。

    行为说明:
        1. 持锁期间抛出异常，随后另一个进程可重新持锁。
    """
    with pytest.raises(ValueError, match="synthetic failure"), application_lock(tmp_path):
        raise ValueError("synthetic failure")
    result = _run_child(tmp_path)
    assert result.returncode == 0, result.stderr


def test_abrupt_process_exit_releases_lock(tmp_path):
    """验证进程来不及运行 finally 时仍能恢复启动。

    作用:
        确认锁由操作系统释放，磁盘上的旧 PID 不会永久阻塞程序。

    参数:
        tmp_path: 隔离目录。

    返回:
        None。

    行为说明:
        1. 子进程持锁后调用 os._exit，父进程随即可以持锁。
    """
    result = _run_child(tmp_path, crash=True)
    assert result.returncode == 7, result.stderr
    with application_lock(tmp_path):
        assert json.loads((tmp_path / ".runtime.lock").read_text())["pid"] == os.getpid()


def test_cli_holds_lock_while_run_executes(tmp_path, monkeypatch):
    """验证 CLI 在运行前取得锁并在返回后释放。

    作用:
        防止只实现锁函数却遗漏实际命令入口。

    参数:
        tmp_path: 隔离目录。
        monkeypatch: 替换 CLI 逻辑和用户数据路径。

    返回:
        None。

    行为说明:
        1. 用替身 CLI 检查执行时锁已占用，并检查会话参数原样传递。
        2. main 返回后可以再次加锁。
    """
    calls = []
    selected = []

    class Service:
        """将核心替换为无网络对象，保留真实程序装配和文件锁。"""

        def __init__(self, *args, **kwargs):
            """接受实际服务参数；本测试不创建模型。

            作用:
                为实际程序入口提供不访问网络的隔离替身。
            参数:
                args: 服务构造的位置参数。
                kwargs: 服务构造或会话选择选项。
            返回:
                None。
            行为说明:
                1. 资源管理仍由被测程序执行，替身不操作个人数据。
            """

        def client(self, source, **kwargs):
            """记录入口会话选择并返回独立客户端。

            作用:
                为实际程序入口提供不访问网络的隔离替身。
            参数:
                source: 入口来源标识。
                kwargs: 服务构造或会话选择选项。
            返回:
                合成客户端。
            行为说明:
                1. 资源管理仍由被测程序执行，替身不操作个人数据。
            """
            selected.append((source, kwargs))
            return SimpleNamespace(close=lambda: None)

        def cancel_all(self):
            """合成服务没有待取消任务。

            作用:
                为实际程序入口提供不访问网络的隔离替身。
            返回:
                None。
            行为说明:
                1. 资源管理仍由被测程序执行，替身不操作个人数据。
            """

        def stop_accepting(self):
            """合成服务不接受新任务。

            作用:
                为实际程序入口提供不访问网络的隔离替身。
            返回:
                None。
            行为说明:
                1. 资源管理仍由被测程序执行，替身不操作个人数据。
            """

        def close(self):
            """合成服务没有外部资源。

            作用:
                为实际程序入口提供不访问网络的隔离替身。
            返回:
                None。
            行为说明:
                1. 资源管理仍由被测程序执行，替身不操作个人数据。
            """

    def fake_run(**kwargs):
        """在 CLI 替身内检查锁状态。

        作用:
            代替交互终端，避免读取个人配置或调用模型。
        参数:
            kwargs: main 传入的会话选项。
        返回:
            None。
        行为说明:
            1. 记录参数，确认第二个锁申请被拒绝。
        """
        calls.append(kwargs)
        with pytest.raises(AppAlreadyRunning):
            with application_lock(tmp_path):
                pytest.fail("CLI 未持有运行锁")

    monkeypatch.setattr(config, "get_app_dir", lambda: tmp_path)
    monkeypatch.setattr(program, "ApplicationService", Service)
    monkeypatch.setattr(program, "Settings", lambda: SimpleNamespace(kairos_api_key="synthetic"))
    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["kairos", "--channels", "terminal", "--session", "example"])
    main()
    assert selected == [("terminal", {"session_id": "example", "resume_latest": False})]
    assert len(calls) == 1 and calls[0]["allow_detach"] is False
    with application_lock(tmp_path):
        pass


def test_occupied_cli_reports_error_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    """验证已占用时拒绝 CLI 启动并明确报告失败。

    作用:
        避免入口占用被误认为正常运行成功。

    参数:
        tmp_path: 隔离目录。
        monkeypatch: 替换 CLI 与参数。
        capsys: 捕获输出。

    返回:
        None。

    行为说明:
        1. 模拟微信持锁，CLI 不执行、退出码为 1，错误不包含数据路径。
    """
    monkeypatch.setattr(config, "get_app_dir", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["kairos"])
    monkeypatch.setattr(cli, "run", lambda **kwargs: pytest.fail("被占用时不应启动 CLI"))
    with application_lock(tmp_path, label="Kairos Weixin"):
        with pytest.raises(SystemExit) as error:
            main()
    assert error.value.code == 1
    output = capsys.readouterr()
    assert "另一个 Kairos 程序" in output.out
    assert str(tmp_path) not in output.out + output.err


def test_cli_help_does_not_create_runtime_lock(tmp_path, monkeypatch, capsys):
    """验证帮助命令不需要占用运行数据。

    作用:
        保持原有 --help 行为，并避免帮助操作创建状态文件。

    参数:
        tmp_path: 隔离目录。
        monkeypatch: 替换数据目录与进程参数。
        capsys: 捕获帮助输出。

    返回:
        None。

    行为说明:
        1. main --help 以状态码 0 退出，数据目录和运行锁均未创建。
    """
    data_dir = tmp_path / "uncreated"
    monkeypatch.setattr(config, "get_app_dir", lambda: data_dir)
    monkeypatch.setattr(sys, "argv", ["kairos", "--help"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 0
    assert "--session" in capsys.readouterr().out
    assert not data_dir.exists()


def test_main_returns_program_failure_as_nonzero_exit(monkeypatch):
    """将统一程序的失败状态转为命令行退出码。

    作用:
        程序层已负责脱敏和锁内收尾，main 不把失败当成正常退出。
    参数:
        monkeypatch: 替换程序返回值与命令行参数。
    返回:
        None。
    行为说明:
        1. 资源失败时的实际持锁关闭在 test_program 中验证。
    """
    monkeypatch.setattr(program, "run", lambda **kwargs: 1)
    monkeypatch.setattr(sys, "argv", ["kairos"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
