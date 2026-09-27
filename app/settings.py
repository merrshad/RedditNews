"""Configuration: environment variables (pydantic-settings) + ``config/topics.yaml``.

Invariant 6: secrets come from the environment only and are never logged.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models import ImportanceLevel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOPICS_PATH = PROJECT_ROOT / "config" / "topics.yaml"


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


class TopicConfig(BaseModel):
    """One allowed topic and the RSS feeds that feed it (NFR-7)."""

    key: str
    name: str
    feeds: list[str] = Field(default_factory=list)


class TopicsConfig(BaseModel):
    topics: list[TopicConfig]

    @model_validator(mode="after")
    def _check_unique_keys(self) -> "TopicsConfig":
        keys = [topic.key for topic in self.topics]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate topic keys in config/topics.yaml")
        if not keys:
            raise ValueError("config/topics.yaml must define at least one topic")
        return self


def load_topics(path: str | Path = DEFAULT_TOPICS_PATH) -> list[TopicConfig]:
    """Read the allowed topics + feeds from YAML (FR-5, NFR-7)."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return TopicsConfig.model_validate(raw).topics


def topic_display_names(topics: list[TopicConfig]) -> dict[str, str]:
    """Map topic key -> Persian display name for Telegram messages (FR-10)."""
    return {topic.key: topic.name for topic in topics}
