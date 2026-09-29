"""Environment configuration (pydantic-settings) — AGENTS.md section 11.

Invariant 6: secrets come from the environment only and are never logged. Topics,
feeds and their per-source batch caps are *data* and live in Postgres (phase 5); only
their global defaults and the bot credentials come from here.
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

    # Telegram (required): the bot talks to two channels and to the admins
    telegram_bot_token: str
    telegram_chat_id: str  # the public broadcast channel
    telegram_review_channel_id: str  # the private channel where posts wait for a human
    # Comma-separated Telegram user ids allowed to press the review buttons and run the
    # admin commands. Empty means "nobody", which is the safe default (FR-12).
    telegram_admin_ids: str = ""

    # Pipeline behaviour
    poll_interval_seconds: int = 900
    rss_fetch_limit: int = 25  # default per-source batch cap; a source may override it
    similarity_lookback_limit: int = 50
    similarity_lookback_hours: int = 72
    min_importance_to_send: ImportanceLevel = "low"
    http_max_retries: int = 3
    log_level: str = "INFO"

    @property
    def admin_ids(self) -> frozenset[str]:
        """The allow-list as a set; ``"1, 2"`` -> ``{"1", "2"}`` (FR-12)."""
        return frozenset(part.strip() for part in self.telegram_admin_ids.split(",") if part.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process from env/.env."""
    return Settings()  # type: ignore[call-arg]  # values come from env/.env
