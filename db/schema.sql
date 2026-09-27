-- The only source of database schema changes (AGENTS.md Invariant 9, section 10).
-- Applied automatically by the postgres container on first boot via
-- /docker-entrypoint-initdb.d/schema.sql (see docker-compose.yml).
CREATE TABLE IF NOT EXISTS posts (
    id                  BIGSERIAL PRIMARY KEY,
    reddit_id           TEXT NOT NULL UNIQUE,        -- RSS feed guid, e.g. t3_1abcde
    subreddit           TEXT NOT NULL,
    source_topic_key    TEXT NOT NULL,               -- source topic key from config/topics.yaml
    title               TEXT NOT NULL,
    url                 TEXT NOT NULL,
    author              TEXT,
    raw_content         TEXT,                        -- raw text/summary from RSS
    published_at        TIMESTAMPTZ,
    fetched_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- LLM analysis output
    is_relevant         BOOLEAN,
    duplicate_of_id     BIGINT REFERENCES posts(id),
    topic               TEXT,                        -- LLM-confirmed topic (from the allowed list)
    importance          TEXT CHECK (importance IN ('low','medium','high')),
    summary_fa          TEXT,
    key_points          JSONB,                       -- ["نکته ۱", "نکته ۲", ...]
    llm_raw_response    JSONB,                       -- raw LLM output for audit/debug

    status              TEXT NOT NULL DEFAULT 'new'
                        CHECK (status IN (
                            'new', 'skipped_irrelevant', 'skipped_duplicate',
                            'skipped_low_importance', 'to_send', 'sent', 'failed'
                        )),
    sent_at             TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_posts_status        ON posts (status);
CREATE INDEX IF NOT EXISTS idx_posts_published_at  ON posts (published_at DESC);
