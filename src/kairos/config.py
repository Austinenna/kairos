"""配置：从当前用户的 ~/.kairos/.env 和环境变量里读取。"""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 开发资料仍可相对项目根目录定位，用户配置和运行数据不依赖它。
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_app_dir() -> Path:
    """返回当前系统用户的 Kairos 数据目录。

    作用:
        为配置、数据库和记忆提供统一路径，切换项目或工作目录时保持一致。
        每次调用都从当前用户主目录计算，不创建目录或读取个人数据。

    返回:
        当前用户主目录下的 .kairos 路径 (Path)。

    行为说明:
        1. 使用 Path.home() 获取当前用户的主目录，不硬编码用户名。
        2. 拼接 .kairos，使其他电脑也使用各自用户目录下的相同布局。
    """
    return Path.home() / ".kairos"


class ToolSettings(BaseSettings):
    """读取可选工具配置，不要求模型或认证字段。"""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __init__(self, **values: Any) -> None:
        """从当前用户配置及环境变量构造运行设置。

        作用:
            在实例化时确定默认配置位置，使运行时与测试可以使用不同的数据目录。
            读取配置并交给 Pydantic 校验，不创建或修改配置文件。

        参数:
            values: 显式设置字段及 BaseSettings 初始化选项。
                _env_file 可指定其他配置文件；传入 None 时禁用配置文件读取。

        返回:
            无返回值，初始化当前设置实例。

        行为说明:
            1. 未指定 _env_file 时，动态读取 get_app_dir() 下的 .env。
            2. 沿用 BaseSettings 的优先级：显式字段、环境变量、配置文件、默认值。
            3. 显式指定 _env_file（包括 None）时，不回退读取默认配置文件。
        """
        if "_env_file" not in values:
            values["_env_file"] = get_app_dir() / ".env"
        super().__init__(**values)

    # 本地工具模块的加载配置；设为 [] 可独立运行 Kairos 核心。
    kairos_tool_factories: list[str] = Field(default_factory=lambda: ["kairos_todo.tools:build_tools"])


class Settings(ToolSettings):
    """在可选工具配置之外读取模型与认证配置。"""

    # 字段名不区分大小写地对应环境变量，例如 kairos_provider <- KAIROS_PROVIDER。
    kairos_provider: Literal["openai", "anthropic"] = "openai"
    # 网关项目 Key 不随请求协议变化。
    kairos_api_key: str | None = None
    gateway_base_url: str = Field(min_length=1)
    kairos_model: str = Field(min_length=1)
    # Anthropic SDK 会自行补上 /v1/messages，因此这里填写网关根地址。
    anthropic_base_url: str | None = None
    anthropic_max_tokens: int = 4096
