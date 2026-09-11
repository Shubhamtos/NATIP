"""Application configuration primitives."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "testing", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Environment-backed application settings.

    Attributes:
        environment: Runtime environment name.
        app_name: Human-readable application name.
        database_url: SQLite database URL for local-first persistence.
        log_level: Logging verbosity level.
        debug: Whether debug behavior is enabled.
        secret_key: Optional externally supplied application secret.
        zerodha_api_key: Optional Zerodha API key supplied externally.
        zerodha_api_secret: Optional Zerodha API secret supplied externally.
        zerodha_token_store_path: Local path for Zerodha token storage.
        yahoo_default_exchange_suffix: Default Yahoo Finance suffix for NSE symbols.
        gemini_api_key: Optional Gemini API key supplied externally.
        gemini_model: Gemini model used for optional reasoning enrichment.
        astro_enabled: Whether experimental astro shadow evidence is generated.
        astro_shadow_only: Astro must remain shadow-only in live decision reports.
        astro_max_score_adjustment: Maximum live score adjustment; must be zero.
        astro_ephemeris_path: Local cached Skyfield/JPL ephemeris path.
        raw_material_enabled: Whether raw-material impact research is enabled.
        raw_material_min_confidence: Minimum confidence for material alerts.
        raw_material_margin_alert_bps: Minimum estimated margin impact for material alerts.
    """

    model_config = SettingsConfigDict(
        env_prefix="NATIP_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
    )

    environment: Environment = Field(default="development")
    app_name: str = Field(default="NATIP")
    database_url: str = Field(default="sqlite:///./natip.db")
    log_level: LogLevel = Field(default="INFO")
    debug: bool = Field(default=False)
    secret_key: SecretStr | None = Field(default=None)
    zerodha_api_key: str | None = Field(default=None)
    zerodha_api_secret: SecretStr | None = Field(default=None)
    zerodha_token_store_path: Path = Field(default=Path(".natip/zerodha_tokens.json"))
    yahoo_default_exchange_suffix: str = Field(default=".NS")
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "NATIP_GEMINI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "GOOGLE_GENERATIVE_AI_API_KEY",
        ),
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        validation_alias=AliasChoices("NATIP_GEMINI_MODEL", "GEMINI_MODEL"),
    )
    astro_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("NATIP_ASTRO_ENABLED", "ASTRO_ENABLED"),
    )
    astro_shadow_only: bool = Field(
        default=True,
        validation_alias=AliasChoices("NATIP_ASTRO_SHADOW_ONLY", "ASTRO_SHADOW_ONLY"),
    )
    astro_max_score_adjustment: float = Field(
        default=0.0,
        validation_alias=AliasChoices(
            "NATIP_ASTRO_MAX_SCORE_ADJUSTMENT", "ASTRO_MAX_SCORE_ADJUSTMENT"
        ),
    )
    astro_ephemeris_path: Path = Field(
        default=PROJECT_ROOT / "data" / "astro" / "de421.bsp",
        validation_alias=AliasChoices("NATIP_ASTRO_EPHEMERIS_PATH", "ASTRO_EPHEMERIS_PATH"),
    )
    raw_material_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("NATIP_RAW_MATERIAL_ENABLED", "RAW_MATERIAL_ENABLED"),
    )
    raw_material_min_confidence: float = Field(
        default=70.0,
        ge=0.0,
        le=100.0,
        validation_alias=AliasChoices(
            "NATIP_RAW_MATERIAL_MIN_CONFIDENCE",
            "RAW_MATERIAL_MIN_CONFIDENCE",
        ),
    )
    raw_material_margin_alert_bps: float = Field(
        default=25.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "NATIP_RAW_MATERIAL_MARGIN_ALERT_BPS",
            "RAW_MATERIAL_MARGIN_ALERT_BPS",
        ),
    )

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        """Validate database URL.

        Args:
            value: Configured database URL.

        Returns:
            The validated database URL.

        Raises:
            ValueError: If the database URL is empty or unsupported.
        """

        if not value.strip():
            raise ValueError("database_url must not be empty")
        if not value.startswith(("sqlite:///", "sqlite+aiosqlite:///")):
            raise ValueError("database_url must be a SQLite URL")
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        """Normalize log level values.

        Args:
            value: Raw log level value.

        Returns:
            Uppercase log level.
        """

        return value.upper()

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, value: str) -> str:
        """Normalize environment values.

        Args:
            value: Raw environment value.

        Returns:
            Lowercase environment.
        """

        return value.lower()

    @model_validator(mode="after")
    def validate_environment_contract(self) -> "Settings":
        """Validate environment-specific configuration constraints.

        Returns:
            The validated settings instance.

        Raises:
            ValueError: If production settings are unsafe.
        """

        if self.environment == "production" and self.debug:
            raise ValueError("debug must be false in production")
        if not self.astro_shadow_only:
            raise ValueError("astro_shadow_only must remain true")
        if self.astro_max_score_adjustment != 0:
            raise ValueError("astro_max_score_adjustment must remain 0")
        return self


AppConfig = Settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings.

    Returns:
        Settings: The configured settings instance.
    """

    return Settings()
