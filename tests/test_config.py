from pathlib import Path

import pytest
from pydantic import ValidationError

from kairos import config
from kairos.config import Settings, get_app_dir


@pytest.fixture(autouse=True)
def clean_settings_environment(monkeypatch):
    """清除配置测试相关的环境变量，使测试仅使用构造的离线数据。

    作用:
        将本文件中的配置优先级测试与调用者的环境隔离，避免使用真实配置值。

    参数:
        monkeypatch: pytest 的临时环境变量替换工具。

    返回:
        无返回值，测试结束后由 pytest 恢复环境。

    行为说明:
        1. 根据 Settings 字段名清除大小写两种常用环境变量名称。
        2. 不读取、打印或修改 HOME。
    """
    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)


def test_get_app_dir_uses_current_user_home_without_creating_it(tmp_path, monkeypatch):
    """验证用户目录在每次调用时计算，且路径查询没有目录创建副作用。

    作用:
        检查跨电脑使用当前用户主目录这一约定，不访问真实主目录。

    参数:
        tmp_path: 当前测试的临时目录。
        monkeypatch: 临时替换 Path.home 的测试工具。

    返回:
        无返回值，通过断言验证路径及副作用。

    行为说明:
        1. 依次模拟两个系统用户，分别检查 .kairos 路径。
        2. 确认单纯查询路径不会创建任何用户目录。
    """
    first_home = tmp_path / "first-user"
    second_home = tmp_path / "second-user"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: first_home))
    assert get_app_dir() == first_home / ".kairos"
    assert not first_home.exists()

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: second_home))
    assert get_app_dir() == second_home / ".kairos"
    assert not second_home.exists()


def test_default_config_path_does_not_depend_on_working_directory(isolated_app_dir, tmp_path, monkeypatch):
    """验证在不同工作目录启动时始终读取用户配置，忽略工作目录中的 .env。

    作用:
        防止从项目外启动 Kairos 时读取到其他项目的配置。

    参数:
        isolated_app_dir: 已替代真实用户目录的临时应用目录。
        tmp_path: 当前测试的临时目录。
        monkeypatch: 临时切换工作目录的测试工具。

    返回:
        无返回值，通过断言验证配置来源。

    行为说明:
        1. 在应用目录和其他工作目录创建可区分的离线配置。
        2. 切换工作目录前后分别创建设置，验证两者都来自应用目录。
    """
    (isolated_app_dir / ".env").write_text(
        "GATEWAY_BASE_URL=https://offline.invalid/v1\nKAIROS_MODEL=user-model\n", encoding="utf-8"
    )
    other_directory = tmp_path / "other-project"
    other_directory.mkdir()
    (other_directory / ".env").write_text("KAIROS_MODEL=wrong-project-model\n", encoding="utf-8")

    before = Settings()
    monkeypatch.chdir(other_directory)
    after = Settings()

    assert before.kairos_model == after.kairos_model == "user-model"
    assert after.gateway_base_url == "https://offline.invalid/v1"


def test_settings_resolves_app_directory_at_instantiation(tmp_path, monkeypatch):
    """验证配置默认路径可在导入模块后替换，不缓存其他用户的数据目录。

    作用:
        确保配置入口在创建 Settings 时计算，便于隔离测试和替换用户目录。

    参数:
        tmp_path: 当前测试的临时目录。
        monkeypatch: 临时替换应用目录函数的测试工具。

    返回:
        无返回值，通过断言验证新目录配置生效。

    行为说明:
        1. 创建包含离线设置的替代应用目录。
        2. 替换目录函数后创建 Settings，验证未使用导入时缓存的路径。
    """
    alternate_app_dir = tmp_path / "alternate-apphome"
    alternate_app_dir.mkdir()
    (alternate_app_dir / ".env").write_text(
        "GATEWAY_BASE_URL=https://offline.invalid/v1\nKAIROS_MODEL=alternate-model\n", encoding="utf-8"
    )
    monkeypatch.setattr(config, "get_app_dir", lambda: alternate_app_dir)
    assert Settings().kairos_model == "alternate-model"


def test_environment_overrides_dotenv_and_explicit_fields_override_environment(isolated_app_dir, monkeypatch):
    """验证环境变量覆盖文件，显式传入的字段继续优先于环境变量。

    作用:
        保持迁移前已有的 BaseSettings 配置优先级。

    参数:
        isolated_app_dir: 隔离的应用目录。
        monkeypatch: 临时设置测试环境变量的工具。

    返回:
        无返回值，通过断言验证覆盖顺序。

    行为说明:
        1. 为文件和环境变量设置不同的离线模型名称。
        2. 分别验证环境变量以及显式初始化字段的优先级。
    """
    (isolated_app_dir / ".env").write_text(
        "GATEWAY_BASE_URL=https://offline.invalid/v1\nKAIROS_MODEL=file-model\n", encoding="utf-8"
    )
    monkeypatch.setenv("KAIROS_MODEL", "environment-model")
    assert Settings().kairos_model == "environment-model"
    assert Settings(kairos_model="explicit-model").kairos_model == "explicit-model"


def test_explicit_env_file_does_not_merge_default_file(isolated_app_dir, tmp_path):
    """验证显式配置文件独立读取，不混入默认文件中的其他字段。

    作用:
        保留调用者通过 _env_file 指定独立测试配置的能力。

    参数:
        isolated_app_dir: 隔离的默认应用目录。
        tmp_path: 存放替代配置文件的临时目录。

    返回:
        无返回值，通过断言验证显式配置和字段默认值。

    行为说明:
        1. 在默认配置中放入替代配置不包含的字段。
        2. 读取替代配置，确认该字段仍使用模型默认值。
    """
    (isolated_app_dir / ".env").write_text("KAIROS_MODEL=default-model\nANTHROPIC_MAX_TOKENS=777\n", encoding="utf-8")
    alternate_file = tmp_path / "alternate.env"
    alternate_file.write_text(
        "GATEWAY_BASE_URL=https://offline.invalid/v1\nKAIROS_MODEL=explicit-file-model\n", encoding="utf-8"
    )
    settings = Settings(_env_file=alternate_file)
    assert settings.kairos_model == "explicit-file-model"
    assert settings.anthropic_max_tokens == 4096


def test_explicit_none_disables_default_env_file(isolated_app_dir):
    """验证 _env_file=None 禁用默认文件，即使默认文件中包含完整设置。

    作用:
        防止禁用配置文件读取的离线测试意外使用用户配置。

    参数:
        isolated_app_dir: 存放离线默认配置的临时应用目录。

    返回:
        无返回值，通过断言验证缺失字段及显式初始化行为。

    行为说明:
        1. 禁用配置文件并省略必填字段，检查校验报告确实缺少字段。
        2. 禁用文件但显式给出字段，检查离线设置可正常创建。
    """
    (isolated_app_dir / ".env").write_text(
        "GATEWAY_BASE_URL=https://offline.invalid/v1\nKAIROS_MODEL=file-model\n", encoding="utf-8"
    )
    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None)
    assert {entry["loc"] for entry in error.value.errors()} == {("gateway_base_url",), ("kairos_model",)}

    settings = Settings(_env_file=None, gateway_base_url="https://offline.invalid/v1", kairos_model="offline-model")
    assert settings.kairos_model == "offline-model"
