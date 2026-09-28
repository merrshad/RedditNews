"""Invariant 11: `AGENTS.md` must stay true about the code it documents.

The document is the source of truth for this project (AGENTS.md, section 14), but nothing
used to check that it *kept* being true: a new module, a new env var, a new status or a
renamed column could land while the document quietly rotted. These tests read the real
artefacts — the file tree, `app.settings.Settings`, `app.models.PostStatus`, `db/schema.sql`
and `config/topics.yaml` — and compare them with sections 9, 10 and 11 of AGENTS.md, so a
mismatch fails the suite instead of waiting for the next audit.

They are deliberately about *structure* (names, sets, columns), not prose: wording stays
free, but the contract between the document and the code does not.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

import pytest

from app import (
    analyzer,
    formatting,
    pipeline,
    reddit_source,
    repository,
    telegram_notifier,
)
from app.models import PostStatus
from app.reddit_source import DEFAULT_TOPICS_PATH, load_topics_config
from app.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENTS_PATH = PROJECT_ROOT / "AGENTS.md"
SCHEMA_PATH = PROJECT_ROOT / "db" / "schema.sql"

SECTION_HEADERS = {
    "structure": "## ۹.",
    "schema": "## ۱۰.",
    "config": "## ۱۱.",
    "conventions": "## ۱۲.",
}

_ANY_HEADER_RE = re.compile(r"^## .*$", re.MULTILINE)


def _agents_text() -> str:
    return AGENTS_PATH.read_text(encoding="utf-8")


def _section(name: str) -> str:
    """The text of one AGENTS.md section, up to the next `## ` header of any kind."""
    text = _agents_text()
    header = SECTION_HEADERS[name]
    start = text.index(header)
    following = [match.start() for match in _ANY_HEADER_RE.finditer(text, start + len(header))]
    return text[start : following[0]] if following else text[start:]


def _fenced_block(section: str, language: str) -> str:
    """The first ```<language> ... ``` block of a section."""
    match = re.search(rf"```{language}\n(.*?)```", section, re.DOTALL)
    assert match, f"no {language} block found"
    return match.group(1)


def _documented_python_files() -> set[str]:
    return set(re.findall(r"\b([A-Za-z0-9_]+\.py)\b", _section("structure")))


def _code_python_files() -> set[str]:
    """Basenames of every Python file in `app/` and `tests/` (unique in this project)."""
    return {
        path.name
        for directory in ("app", "tests")
        for path in (PROJECT_ROOT / directory).rglob("*.py")
        if "__pycache__" not in path.parts
    }


def _create_table_block(sql: str) -> str:
    """The body of `CREATE TABLE IF NOT EXISTS posts ( ... );`."""
    match = re.search(r"CREATE TABLE IF NOT EXISTS posts \((.*?)\n\);", sql, re.DOTALL)
    assert match, "the posts table was not found"
    return match.group(1)


def _column_names(create_table_body: str) -> set[str]:
    """Column names of a CREATE TABLE body (comment and constraint lines are skipped)."""
    return set(re.findall(r"^\s{4}([a-z_][a-z0-9_]*)\s", create_table_body, re.MULTILINE))


def _statuses_in(sql: str) -> set[str]:
    """The values listed in the `status ... CHECK (status IN (...))` constraint."""
    match = re.search(r"status IN \((.*?)\)", sql, re.DOTALL)
    assert match, "no status CHECK constraint found"
    return set(re.findall(r"'([a-z_]+)'", match.group(1)))


def _documented_env_vars() -> set[str]:
    return set(re.findall(r"^\| `([A-Z_]+)` \|", _section("config"), re.MULTILINE))


# --- section 9: the file tree ------------------------------------------------------


def test_every_module_and_test_file_is_listed_in_the_documented_tree() -> None:
    assert _code_python_files() - _documented_python_files() == set()


def test_the_documented_tree_lists_no_file_that_does_not_exist() -> None:
    assert _documented_python_files() - _code_python_files() == set()


def test_the_prompt_template_is_documented() -> None:
    """The prompt lives outside the code on purpose, so the tree must point at it."""
    assert "analysis_prompt.md" in _section("structure")
    assert (PROJECT_ROOT / "app" / "prompts" / "analysis_prompt.md").is_file()


@pytest.mark.parametrize(
    ("module", "attribute"),
    [
        (reddit_source, "load_topics_config"),
        (reddit_source, "fetch_all"),
        (reddit_source, "topic_display_names"),
        (repository, "exists"),
        (repository, "save"),
        (repository, "update_status"),
        (repository, "fetch_recent_candidates"),
        (repository, "fetch_pending_to_send"),
        (analyzer, "analyze"),
        (telegram_notifier, "send_message"),
        (formatting, "format_message"),
        (pipeline, "run_once"),
        (pipeline, "retry_pending_sends"),
    ],
)
def test_the_apis_named_in_section_9_are_the_real_ones(module: object, attribute: str) -> None:
    """A documented public API must exist, and the public APIs must be documented."""
    assert hasattr(module, attribute)
    assert attribute in _section("structure")


# --- section 10: the schema --------------------------------------------------------


def test_the_posts_columns_are_documented_exactly() -> None:
    real = _column_names(_create_table_block(SCHEMA_PATH.read_text(encoding="utf-8")))
    documented = _column_names(_create_table_block(_fenced_block(_section("schema"), "sql")))

    assert documented == real
    assert len(real) == 20  # guards against the regex silently matching nothing


def test_the_documented_statuses_are_the_real_ones() -> None:
    """`PostStatus` (code), `db/schema.sql` (database) and section 10 (doc) must agree."""
    real = set(get_args(PostStatus))
    schema = _statuses_in(SCHEMA_PATH.read_text(encoding="utf-8"))
    documented = _statuses_in(_fenced_block(_section("schema"), "sql"))

    assert schema == real
    assert documented == real


def test_the_documented_indexes_are_the_real_ones() -> None:
    documented = set(re.findall(r"idx_posts_[a-z_]+", _fenced_block(_section("schema"), "sql")))
    real = set(re.findall(r"idx_posts_[a-z_]+", SCHEMA_PATH.read_text(encoding="utf-8")))

    assert documented == real
    assert documented  # the regex really found the two indexes


# --- section 11: configuration -----------------------------------------------------


def test_every_setting_is_documented_and_every_documented_var_exists() -> None:
    """NFR-7: the document is how an operator learns which knobs exist.

    Compared upper-case because that is how the env names are written and how
    pydantic-settings maps them onto the lower-case fields (``case_sensitive=False``).
    """
    assert _documented_env_vars() == {name.upper() for name in Settings.model_fields}


def test_the_env_example_matches_the_documented_variables() -> None:
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    names = set(re.findall(r"^([A-Z_]+)=", example, re.MULTILINE))

    assert names == _documented_env_vars()
    # `cp .env.example .env` must produce a readable file: every non-comment line has to be
    # a KEY=value pair, or whatever else is in there gets copied into a live configuration.
    assert all(
        line.strip() == "" or line.startswith("#") or "=" in line
        for line in example.splitlines()
    ), "every non-comment line of .env.example must be a KEY=value pair"


def test_the_documented_topic_list_is_the_shipped_one() -> None:
    """Section 11 shows `config/topics.yaml`; the example must not be a stale sample."""
    block = _fenced_block(_section("config"), "yaml")
    documented_keys = set(re.findall(r"key: ([a-z0-9_-]+)", block))
    documented_feeds = set(re.findall(r'"(https://[^"]+)"', block))

    real = load_topics_config(DEFAULT_TOPICS_PATH)
    real_keys = {topic.key for topic in real.topics}
    real_feeds = {feed for topic in real.topics for feed in topic.feeds}

    assert real_keys  # the shipped file is not empty
    assert documented_keys == real_keys
    assert documented_feeds == real_feeds


@pytest.mark.parametrize("header", list(SECTION_HEADERS.values()))
def test_the_documented_sections_exist(header: str) -> None:
    """Cheap guard: renaming a section breaks every check above in a confusing way."""
    assert header in _agents_text()
