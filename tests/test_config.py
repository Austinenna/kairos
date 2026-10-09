from kairos.config import Settings


def test_anthropic_settings_can_reuse_gateway_deepseek_credentials(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "KAIROS_PROVIDER=anthropic\n"
        "KAIROS_MODEL=deepseek-flash\n"
        "DEEPSEEK_API_KEY=gateway-token\n"
        "DEEPSEEK_BASE_URL=https://gateway.example/v1\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.kairos_api_key == "gateway-token"
    assert settings.openai_base_url == "https://gateway.example/v1"


def test_generic_kairos_key_takes_precedence_over_legacy_alias(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "KAIROS_API_KEY=canonical-token\nDEEPSEEK_API_KEY=legacy-token\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.kairos_api_key == "canonical-token"
