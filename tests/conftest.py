from pathlib import Path

import pytest

from kairos import config


@pytest.fixture(autouse=True)
def isolated_app_dir(tmp_path, monkeypatch) -> Path:
    """将每个测试的 Kairos 用户数据目录隔离到临时目录。

    作用:
        阻止测试读取真实用户配置或创建真实数据库、记忆文件。
        测试仍然通过正常路径入口取得目录，不修改系统 HOME 环境变量。

    参数:
        tmp_path: pytest 为当前测试创建的独立临时目录。
        monkeypatch: pytest 的临时属性替换工具，测试结束后自动恢复。

    返回:
        当前测试独享的 Kairos 数据目录 (Path)。

    行为说明:
        1. 在临时目录中创建 apphome。
        2. 替换 kairos.config.get_app_dir，使默认路径的测试也不会访问真实用户数据。
    """
    app_dir = tmp_path / "apphome"
    app_dir.mkdir()
    monkeypatch.setattr(config, "get_app_dir", lambda: app_dir)
    return app_dir
