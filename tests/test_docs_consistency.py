"""Invariant 11: `AGENTS.md` must stay true about the code it documents.

The document is the source of truth for this project (AGENTS.md, section 14), but nothing
used to check that it *kept* being true: a new module, a new env var, a new status or a
renamed column could land while the document quietly rotted. These tests read the real
artefacts — the file tree, `app.settings.Settings`, `app.models.PostStatus`/`ReviewStatus`,
`db/schema.sql` (columns *and* the seeded taxonomy) and `.env.example` — and compare them
with sections 9, 10 and 11 of AGENTS.md, so a mismatch fails the suite instead of waiting
for the next audit.

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
    review,
    telegram_notifier,
    telegram_updates,
)
from app.models import PostStatus, ReviewStatus
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


def _create_table_block(sql: str, table: str) -> str:
    """The body of `CREATE TABLE IF NOT EXISTS <table> ( ... );`."""
    match = re.search(rf"CREATE TABLE IF NOT EXISTS {table} \((.*?)\n\);", sql, re.DOTALL)
    assert match, f"the {table} table was not found"
    return match.group(1)


def _column_names(create_table_body: str) -> set[str]:
    """Column names of a CREATE TABLE body (comment and constraint lines are skipped)."""
    return set(re.findall(r"^\s{4}([a-z_][a-z0-9_]*)\s", create_table_body, re.MULTILINE))


def _checked_values(sql: str, column: str) -> set[str]:
    """The values of one `CHECK (<column> IN (...))` constraint, wherever it appears."""
    match = re.search(rf"CHECK \({column} IN \((.*?)\)", sql, re.DOTALL)
    assert match, f"no CHECK constraint for {column} found"
    return set(re.findall(r"'([a-z_]+)'", match.group(1)))


def _documented_env_vars() -> set[str]:
    return set(re.findall(r"^\| `([A-Z_]+)` \|", _section("config"), re.MULTILINE))


def _seeded_taxonomy(sql: str) -> tuple[set[str], set[str]]:
    """The topic keys and feed URLs `db/schema.sql` seeds on a fresh database."""
    topics_block = re.search(
        r"INSERT INTO topics.*?ON CONFLICT", sql, re.DOTALL
    )
    sources_block = re.search(r"INSERT INTO sources.*?ON CONFLICT", sql, re.DOTALL)
    assert topics_block and sources_block, "the taxonomy seed was not found in db/schema.sql"

    keys = set(re.findall(r"\('([a-z0-9_-]+)',", topics_block.group(0)))
    urls = set(re.findall(r"'(https://[^']+)'", sources_block.group(0)))
    return keys, urls


# --- section 9: the file tree ------------------------------------------------------


def test_every_module_and_test_file_is_listed_in_the_documented_tree() -> None:
    assert _code_python_files() - _documented_python_files() == set()


def test_the_documented_tree_lists_no_file_that_does_not_exist() -> None:
    assert _documented_python_files() - _code_python_files() == set()


def test_the_prompt_template_is_documented() -> None:
    """The prompt lives outside the code on purpose, so the tree must point at it."""
    assert "analysis_prompt.md" in _section("structure")
    assert (PROJECT_ROOT / "app" / "prompts" / "analysis_prompt.md").is_file()


def test_the_config_directory_is_gone() -> None:
    """Topics and sources are data now (phase 5): no YAML file may come back."""
    assert "topics.yaml" not in _section("structure")
    assert not (PROJECT_ROOT / "config").exists()


@pytest.mark.parametrize(
    ("module", "attribute"),
    [
        (reddit_source, "fetch_all"),
        (reddit_source, "topic_display_names"),
        (repository, "list_topics"),
        (repository, "list_sources"),
        (repository, "exists"),
        (repository, "save"),
        (repository, "fetch_new_reviews"),
        (repository, "mark_review_dispatched"),
        (repository, "decide_review"),
        (repository, "fetch_approved_for_analysis"),
        (repository, "record_analysis"),
        (repository, "fetch_pending_to_send"),
        (repository, "claim_for_publish"),
        (repository, "mark_published"),
        (repository, "fetch_recent_candidates"),
        (repository, "find_unusable_tables"),
        (analyzer, "analyze"),
        (telegram_notifier, "send_message"),
        (telegram_notifier, "edit_message_text"),
        (telegram_notifier, "answer_callback_query"),
        (telegram_notifier, "get_updates"),
        (review, "dispatch_pending_reviews"),
        (review, "handle_callback"),
        (telegram_updates, "poll_once"),
        (formatting, "format_message"),
        (pipeline, "run_once"),
        (pipeline, "process_approved_posts"),
        (pipeline, "retry_pending_sends"),
    ],
)
def test_the_apis_named_in_section_9_are_the_real_ones(module: object, attribute: str) -> None:
    """A documented public API must exist, and the public APIs must be documented."""
    assert hasattr(module, attribute)
    assert attribute in _section("structure")


# --- section 10: the schema --------------------------------------------------------


@pytest.mark.parametrize("table", ["topics", "sources", "posts"])
def test_the_table_columns_are_documented_exactly(table: str) -> None:
    real = _column_names(_create_table_block(SCHEMA_PATH.read_text(encoding="utf-8"), table))
    documented = _column_names(
        _create_table_block(_fenced_block(_section("schema"), "sql"), table)
    )

    assert documented == real
    assert real, f"the {table} regex matched nothing"


def test_the_posts_columns_count_is_guarded() -> None:
    """A canary for the regex above: the review/audit columns are really there."""
    real = _column_names(_create_table_block(SCHEMA_PATH.read_text(encoding="utf-8"), "posts"))

    assert len(real) == 31


@pytest.mark.parametrize(
    ("column", "literal"),
    [("status", PostStatus), ("review_status", ReviewStatus)],
)
def test_the_documented_states_are_the_real_ones(column: str, literal: object) -> None:
    """The literal (code), `db/schema.sql` (database) and section 10 (doc) must agree."""
    real = set(get_args(literal))
    schema = _checked_values(SCHEMA_PATH.read_text(encoding="utf-8"), column)
    documented = _checked_values(_fenced_block(_section("schema"), "sql"), column)

    assert schema == real
    assert documented == real


def test_the_documented_indexes_are_the_real_ones() -> None:
    documented = set(re.findall(r"idx_[a-z_]+", _fenced_block(_section("schema"), "sql")))
    real = set(re.findall(r"idx_[a-z_]+", SCHEMA_PATH.read_text(encoding="utf-8")))

    assert documented == real
    assert documented  # the regex really found the indexes


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


def test_the_seeded_taxonomy_is_the_documented_one() -> None:
    """Section 11 shows the starting topics/feeds; the database seed must not drift."""
    topics_block = _fenced_block(_section("config"), "sql")
    documented_keys = set(re.findall(r"\('([a-z0-9_-]+)',", topics_block))
    documented_urls = set(re.findall(r"'(https://[^']+)'", topics_block))

    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    real_keys, real_urls = _seeded_taxonomy(schema)

    assert real_keys and real_urls  # the shipped seed is not empty
    assert documented_keys == real_keys
    assert documented_urls == real_urls


@pytest.mark.parametrize("header", list(SECTION_HEADERS.values()))
def test_the_documented_sections_exist(header: str) -> None:
    """Cheap guard: renaming a section breaks every check above in a confusing way."""
    assert header in _agents_text()
