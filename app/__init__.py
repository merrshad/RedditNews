"""reddit-telegram-digest — RSS -> human review -> one LLM analysis -> Telegram channels.

Module responsibilities (AGENTS.md section 9):
- settings:     environment/config loading
- models:       Pydantic models (LLM output validation lives here)
- reddit_source: RSS fetching/parsing with a per-source cap   (FR-1)
- repository:   all database access, including the atomic state transitions
                (FR-2..FR-15)
- llm_client:   thin OpenAI-compatible wrapper   (NFR-2)
- analyzer:     prompt building + output parsing (FR-3..FR-8)
- review:       the human gate: private-channel dispatch + ✅/❌ decisions (FR-12)
- telegram_updates: Telegram update intake (cursor + routing)  (FR-12)
- formatting:   PostRecord -> Persian Telegram text (FR-10)
- telegram_notifier: the whole Telegram Bot API boundary (FR-10, FR-12)
- pipeline:     orchestration of the above       (FR-11, FR-13)
- main:         long-poll + scheduled cycle entry point
"""
