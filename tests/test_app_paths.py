"""验证默认用户数据路径与显式测试路径隔离。"""

import os
import stat

from kairos import config
from kairos.runtime import create_runtime


class OfflineModel:
    model = "offline-path-check"

    def complete(self, *, messages, tools):
        return {"content": "离线回答"}


def test_default_database_is_shared_across_working_directories(tmp_path, monkeypatch):
    """验证切换启动目录仍然恢复同一用户数据库。

    作用:
        检查全局路径解析不依赖项目 cwd，测试只操作隔离的临时目录。

    参数:
        tmp_path: 测试目录。
        monkeypatch: 临时切换 cwd。

    返回:
        None。

    行为说明:
        1. 在两个不同 cwd 创建并恢复同一会话，检查默认数据库位置和身份。
    """
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    monkeypatch.chdir(first_dir)
    with create_runtime(model=OfflineModel()) as runtime:
        session_id = runtime.session_id
        runtime.rename_session("可跨目录恢复")
    monkeypatch.chdir(second_dir)
    with create_runtime(model=OfflineModel(), session_id=session_id) as restored:
        assert restored.session_id == session_id
        assert restored.session_info.title == "可跨目录恢复"
        database = restored.connection.execute("PRAGMA database_list").fetchone()[2]
        assert database == str(config.get_app_dir() / "kairos.sqlite3")
    assert not (first_dir / "data").exists()
    assert not (second_dir / "data").exists()
    if os.name == "posix":
        assert stat.S_IMODE((config.get_app_dir() / "kairos.sqlite3").stat().st_mode) == 0o600


def test_explicit_database_path_still_overrides_default(tmp_path):
    """验证调用方仍可显式提供隔离数据库。

    作用:
        保持运行时测试和嵌入接口的路径注入能力。

    参数:
        tmp_path: 测试目录。

    返回:
        None。

    行为说明:
        1. 指定数据库后确认连接使用该文件，默认数据库没有被创建。
    """
    selected = tmp_path / "selected.sqlite3"
    with create_runtime(model=OfflineModel(), db_path=selected) as runtime:
        assert runtime.connection.execute("PRAGMA database_list").fetchone()[2] == str(selected)
    assert not (config.get_app_dir() / "kairos.sqlite3").exists()
