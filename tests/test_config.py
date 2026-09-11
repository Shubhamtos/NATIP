import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_settings_supports_development_environment() -> None:
    settings = Settings(environment="development", database_url="sqlite:///./dev.db")

    assert settings.environment == "development"
    assert settings.database_url == "sqlite:///./dev.db"


def test_settings_supports_testing_environment() -> None:
    settings = Settings(environment="testing", database_url="sqlite:///./test.db")

    assert settings.environment == "testing"


def test_settings_supports_production_environment_without_debug() -> None:
    settings = Settings(environment="production", debug=False, database_url="sqlite:///./prod.db")

    assert settings.environment == "production"
    assert settings.debug is False


def test_settings_rejects_unknown_environment() -> None:
    with pytest.raises(ValidationError):
        Settings(environment="local")


def test_settings_rejects_non_sqlite_database_url() -> None:
    with pytest.raises(ValidationError):
        Settings(database_url="postgresql://example")


def test_settings_rejects_debug_in_production() -> None:
    with pytest.raises(ValidationError):
        Settings(environment="production", debug=True)


def test_settings_normalizes_log_level() -> None:
    settings = Settings(log_level="debug")

    assert settings.log_level == "DEBUG"


def test_settings_supports_gemini_configuration() -> None:
    settings = Settings(
        gemini_api_key="test-secret",
        gemini_model="gemini-test-model",
    )

    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == "test-secret"
    assert settings.gemini_model == "gemini-test-model"


def test_settings_supports_common_gemini_env_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NATIP_GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "alias-secret")
    monkeypatch.setenv("GEMINI_MODEL", "alias-model")

    settings = Settings(_env_file=None)

    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == "alias-secret"
    assert settings.gemini_model == "alias-model"
