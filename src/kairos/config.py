"""配置：从项目根目录的 .env 和环境变量里读取。"""

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/kairos/config.py -> 项目根目录。这样无论在哪个目录运行，都能找到 .env
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 字段名不区分大小写地对应环境变量，例如 kairos_provider <- KAIROS_PROVIDER。
    kairos_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "DEEPSEEK_API_KEY"),
    )
    openai_base_url: str = Field(
        default="https://api.deepseek.com",
        validation_alias=AliasChoices("OPENAI_BASE_URL", "DEEPSEEK_BASE_URL"),
    )
    kairos_model: str = "deepseek-flash"
    # 网关项目 Token 可同时用于 OpenAI 兼容和 Anthropic Messages 协议。
    # 保留明确的 ANTHROPIC_API_KEY，同时兼容当前项目已有的 key 命名。
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"),
    )
    # Anthropic SDK 会自行补上 /v1/messages，因此这里填写网关根地址。
    anthropic_base_url: str | None = Field(default=None, validation_alias=AliasChoices("ANTHROPIC_BASE_URL"))
    anthropic_max_tokens: int = 4096
