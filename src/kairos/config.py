"""配置：从环境变量和 ~/.kairos/.env 读取。

字段名的大写就是环境变量名，例如 kairos_model ← KAIROS_MODEL。
读取优先级：代码里显式传入 > 环境变量 > ~/.kairos/.env > 这里写的默认值。
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_app_dir() -> Path:
    """Kairos 的数据目录：配置、数据库、记忆都放在这里，跟随当前系统用户。"""
    return Path.home() / ".kairos"


def get_workspace_dir() -> Path:
    """工作目录：文件工具唯一能写入的地方（写入白名单），放模型生成和修改的文件。"""
    return Path.home() / "Documents" / "kairos"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file_encoding="utf-8", extra="ignore")

    def __init__(self, **values):
        # 每次创建时才确定 .env 的位置（测试会把数据目录换成临时目录）；传 _env_file=None 表示不读文件。
        values.setdefault("_env_file", get_app_dir() / ".env")
        super().__init__(**values)

    kairos_provider: Literal["openai", "anthropic"] = "openai"  # 请求协议
    kairos_api_key: str | None = None  # 网关的 Key，两种协议共用
    gateway_base_url: str = Field(min_length=1)  # 必填
    kairos_model: str = Field(min_length=1)  # 必填
    anthropic_base_url: str | None = None  # 不填时用 gateway_base_url 去掉末尾的 /v1
    anthropic_max_tokens: int = 4096
    # 要加载的业务工具模块，格式 "模块:函数"。不配置时加载待办；设为 [] 不加载任何业务工具。
    kairos_tool_factories: list[str] = ["kairos_todo.tools:build_tools"]
