"""reddit-telegram-digest — RSS -> single LLM analysis -> Postgres -> Telegram broadcast.

Module responsibilities (AGENTS.md section 9):
- settings:     environment/config loading
- models:       Pydantic models (LLM output validation lives here)
- reddit_source: RSS fetching/parsing            (FR-1)
- repository:   all database access              (FR-2, FR-9, FR-11)
- llm_client:   thin OpenAI-compatible wrapper   (NFR-2)
- analyzer:     prompt building + output parsing (FR-3..FR-8)
- formatting:   PostRecord -> Persian Telegram text (FR-10)
- telegram_notifier: Telegram Bot API sendMessage (FR-10)
- pipeline:     orchestration of the above
- main:         periodic entry point
"""
