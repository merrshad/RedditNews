"""FR-10 / Invariant 5 tests: pure formatting, no I/O.

The exact message layout is the contract documented in AGENTS.md section 12.
"""

from __future__ import annotations

import pytest

from app.formatting import (
    IMPORTANCE_LABELS_FA,
    MISSING_SUMMARY_LABEL,
    TELEGRAM_MESSAGE_LIMIT,
    UNKNOWN_IMPORTANCE_LABEL,
    UNKNOWN_TOPIC_LABEL,
    escape_html,
    format_message,
    importance_label,
)
from app.models import PostRecord

POST_URL = "https://www.reddit.com/r/MachineLearning/comments/1abcde/x/"


def _post(**overrides: object) -> PostRecord:
    data = {
        "id": 42,
        "reddit_id": "t3_1abcde",
        "subreddit": "MachineLearning",
        "source_topic_key": "ai",
        "title": "A new open model was released",
        "url": POST_URL,
        "author": "somebody",
        "topic": "ai",
        "importance": "high",
        "summary_fa": "این یک خلاصه فارسی است.",
        "key_points": ["نکته اول", "نکته دوم"],
        "status": "to_send",
    }
    data.update(overrides)
    return PostRecord(**data)


# --- the message itself (FR-10) ---------------------------------------------------


def test_format_message_contains_every_required_section() -> None:
    message = format_message(_post(), topic_name="هوش مصنوعی")

    assert "📌 <b>A new open model was released</b>" in message
    assert "r/MachineLearning • هوش مصنوعی • اهمیت: بالا" in message
    assert "این یک خلاصه فارسی است." in message
    assert "🔑 نکات کلیدی:" in message
    assert "• نکته اول" in message
    assert "• نکته دوم" in message
    assert message.endswith(f"🔗 {POST_URL}")


def test_format_message_keeps_the_documented_section_order() -> None:
    message = format_message(_post())

    assert message.index("📌") < message.index("این یک خلاصه فارسی است.")
    assert message.index("این یک خلاصه فارسی است.") < message.index("🔑")
    assert message.index("🔑") < message.index("🔗")


def test_format_message_falls_back_to_the_stored_topic_key() -> None:
    """Without a Persian display name the key is still better than nothing."""
    message = format_message(_post())

    assert "r/MachineLearning • ai • اهمیت: بالا" in message


def test_format_message_uses_a_placeholder_topic_instead_of_a_raw_key() -> None:
    message = format_message(_post(topic=None, source_topic_key="startup"))

    assert UNKNOWN_TOPIC_LABEL in message
    assert "startup" not in message


def test_format_message_escapes_html_in_user_content() -> None:
    message = format_message(
        _post(title="<script>alert(1)</script> & done>", url="https://example.com/?a=1&b=2")
    )

    assert "<script>" not in message
    assert "&lt;script&gt;alert(1)&lt;/script&gt; &amp; done&gt;" in message
    assert "https://example.com/?a=1&amp;b=2" in message


def test_format_message_leaves_quotes_alone() -> None:
    """`html.escape(..., quote=False)` — the text never enters an HTML attribute."""
    message = format_message(_post(title='He said "go"'))

    assert 'He said "go"' in message
    assert "&quot;" not in message


# --- optional sections -------------------------------------------------------------


def test_format_message_omits_the_key_points_section_when_there_are_none() -> None:
    message = format_message(_post(key_points=[]))

    assert "🔑" not in message
    assert "نکات کلیدی" not in message


def test_format_message_ignores_blank_key_points() -> None:
    message = format_message(_post(key_points=["", "   "]))

    assert "🔑" not in message


def test_format_message_uses_a_placeholder_for_a_missing_summary() -> None:
    message = format_message(_post(summary_fa=None))

    assert MISSING_SUMMARY_LABEL in message


# --- importance labels (FR-10) ---------------------------------------------------


@pytest.mark.parametrize(
    ("importance", "expected"), [("low", "کم"), ("medium", "متوسط"), ("high", "بالا")]
)
def test_importance_label_maps_every_known_value(importance: str, expected: str) -> None:
    assert importance_label(importance) == expected


@pytest.mark.parametrize("importance", [None, "", "nonsense"])
def test_importance_label_falls_back_to_the_unknown_label(importance: str | None) -> None:
    assert importance_label(importance) == UNKNOWN_IMPORTANCE_LABEL


def test_importance_label_covers_every_value_of_the_mapping_table() -> None:
    assert {importance_label(value) for value in IMPORTANCE_LABELS_FA} == {"کم", "متوسط", "بالا"}


def test_format_message_reports_a_missing_importance() -> None:
    message = format_message(_post(importance=None, key_points=[]))

    assert f"اهمیت: {UNKNOWN_IMPORTANCE_LABEL}" in message


# --- Telegram's length limit (FR-10) ---------------------------------------------


def test_escape_html_handles_the_three_telegram_entities() -> None:
    assert escape_html("a & b < c > d") == "a &amp; b &lt; c &gt; d"
    assert escape_html('"quoted"') == '"quoted"'


def test_format_message_shortens_the_summary_to_fit_the_limit() -> None:
    message = format_message(_post(summary_fa="ب" * 8000))

    assert len(message) <= TELEGRAM_MESSAGE_LIMIT
    assert "…" in message
    # The shortened summary must not swallow the sections below it.
    assert message.startswith("📌 <b>A new open model was released</b>")
    assert message.endswith(f"🔗 {POST_URL}")


def test_format_message_keeps_a_long_title_and_summary_inside_the_limit() -> None:
    message = format_message(_post(title="ت" * 5000, summary_fa="ب" * 5000))

    assert len(message) <= TELEGRAM_MESSAGE_LIMIT


def test_format_message_survives_a_pathological_key_point_list() -> None:
    message = format_message(
        _post(summary_fa="ب" * 5000, key_points=["ن" * 1000] * 40, title="ت" * 1000)
    )

    assert len(message) <= TELEGRAM_MESSAGE_LIMIT
