"""FR-10 / Invariant 5 tests: pure formatting, no I/O."""

from __future__ import annotations

from app.formatting import (
    IMPORTANCE_LABELS_FA,
    SAFE_MESSAGE_LIMIT,
    UNKNOWN_TOPIC_LABEL,
    escape_html,
    format_post_message,
    importance_label,
)
from app.models import PostRecord


def _post(**overrides: object) -> PostRecord:
    data = {
        "id": 42,
        "reddit_id": "t3_1abcde",
        "subreddit": "MachineLearning",
        "source_topic_key": "ai",
        "title": "A new open model was released",
        "url": "https://www.reddit.com/r/MachineLearning/comments/1abcde/x/",
        "author": "somebody",
        "topic": "ai",
        "importance": "high",
        "summary_fa": "این یک خلاصه فارسی است.",
        "key_points": ["نکته اول", "نکته دوم"],
        "status": "to_send",
    }
    data.update(overrides)
    return PostRecord(**data)


def test_format_post_message_contains_every_required_section() -> None:
    message = format_post_message(_post(), topic_name="هوش مصنوعی")

    assert "A new open model was released" in message
    assert "r/MachineLearning" in message
    assert "هوش مصنوعی" in message
    assert "زیاد" in message  # high -> Persian label
    assert "این یک خلاصه فارسی است." in message
    assert "• نکته اول" in message
    assert "• نکته دوم" in message
    assert '<a href="https://www.reddit.com/r/MachineLearning/comments/1abcde/x/">' in message


def test_format_post_message_falls_back_to_a_placeholder_topic() -> None:
    """A missing Persian topic name must never surface a raw English key."""
    message = format_post_message(_post(topic=None, source_topic_key="startup"))

    assert UNKNOWN_TOPIC_LABEL in message
    assert "startup" not in message


def test_format_post_message_escapes_html_in_user_content() -> None:
    message = format_post_message(
        _post(title="<script>alert(1)</script>", summary_fa="a & b < c > d")
    )

    assert "<script>" not in message
    assert "&lt;script&gt;" in message
    assert "a &amp; b &lt; c &gt; d" in message


def test_format_post_message_uses_a_placeholder_for_a_missing_summary() -> None:
    message = format_post_message(_post(summary_fa=None, key_points=[]))

    assert "خلاصه‌ای برای این پست ثبت نشده است." in message
    assert "🔑" not in message


def test_format_post_message_stays_within_telegram_limits() -> None:
    message = format_post_message(
        _post(summary_fa="ب" * 5000, key_points=["ن" * 1000] * 40, title="ت" * 1000)
    )

    assert len(message) <= SAFE_MESSAGE_LIMIT
    assert message.endswith("…")


def test_format_post_message_truncates_a_very_long_summary() -> None:
    message = format_post_message(_post(summary_fa="ب" * 5000))

    assert message.endswith("مشاهده پست اصلی</a>") or "مشاهده پست اصلی" in message
    assert len(message) < 5000


def test_format_post_message_keeps_the_link_when_the_summary_is_long() -> None:
    message = format_post_message(_post(summary_fa="ب" * 1400))

    assert "مشاهده پست اصلی" in message


def test_escape_html_handles_the_three_telegram_entities() -> None:
    assert escape_html("a & b < c > d") == "a &amp; b &lt; c &gt; d"


def test_importance_label_covers_all_values_and_unknown_input() -> None:
    assert {importance_label(value) for value in IMPORTANCE_LABELS_FA} == {"کم", "متوسط", "زیاد"}
    assert importance_label(None) == "کم"
    assert importance_label("nonsense") == "کم"
