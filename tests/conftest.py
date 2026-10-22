from pathlib import Path

import pytest

from kairos import config


@pytest.fixture(autouse=True)
def isolated_app_dir(tmp_path, monkeypatch) -> Path:
    """每个测试都把 Kairos 数据目录换成临时目录，不会读写真实的 ~/.kairos。"""
    app_dir = tmp_path / "apphome"
    app_dir.mkdir()
    monkeypatch.setattr(config, "get_app_dir", lambda: app_dir)
    return app_dir
