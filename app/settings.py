"""Environment configuration (pydantic-settings) — AGENTS.md section 11.

Invariant 6: secrets come from the environment only and are never logged. The topic
and feed configuration lives next to the RSS source that consumes it
(``app/reddit_source.py``), not here.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models import ImportanceLevel


class Settings(BaseSettings):
    """Typed view over the env vars documented in AGENTS.md section 11."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database (required)
    database_url: str

    # LLM, OpenAI-compatible (required, provider-dependent)
    openai_api_key: str
    openai_model: str
    openai_base_url: str = "https://api.openai.com/v1"

    # Telegram (required)
    telegram_bot_token: str
    telegram_chat_id: str

    # Pipeline behaviour
    poll_interval_seconds: int = 900
    similarity_lookback_limit: int = 50
    similarity_lookback_hours: int = 72
    min_importance_to_send: ImportanceLevel = "low"
    http_max_retries: int = 3
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process from env/.env."""
    return Settings()  # type: ignore[call-arg]  # values come from env/.env
