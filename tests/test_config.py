from kairos.config import Settings


def test_settings_loads_generic_gateway_credentials(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "KAIROS_PROVIDER=anthropic\n"
        "KAIROS_MODEL=deepseek-flash\n"
        "KAIROS_API_KEY=gateway-token\n"
        "DEEPSEEK_BASE_URL=https://gateway.example/v1\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.kairos_api_key == "gateway-token"
    assert settings.openai_base_url == "https://gateway.example/v1"
