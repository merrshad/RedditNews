-- The only source of database schema changes (AGENTS.md Invariant 9, section 10).
-- Applied automatically by the postgres container on first boot via
-- /docker-entrypoint-initdb.d/schema.sql (see docker-compose.yml).

-- Topics are data, not code (phase 5): the admin creates/edits/deletes them through the
-- Telegram admin commands, and `sources` hangs off them.
CREATE TABLE IF NOT EXISTS topics (
    id          BIGSERIAL PRIMARY KEY,
    key         TEXT NOT NULL UNIQUE,                -- the value the LLM may return as `topic` (FR-5)
    name        TEXT NOT NULL,                       -- Persian label shown in Telegram (Invariant 5)
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,       -- disabled topics are not fetched/proposed
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One Reddit RSS feed, its topic and its own batch cap. `fetch_limit` is per source;
-- NULL means "use the RSS_FETCH_LIMIT default" (phase 5, NFR-7).
CREATE TABLE IF NOT EXISTS sources (
    id          BIGSERIAL PRIMARY KEY,
    topic_id    BIGINT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    rss_url     TEXT NOT NULL UNIQUE,
    fetch_limit INTEGER CHECK (fetch_limit IS NULL OR fetch_limit > 0),
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sources_topic_id ON sources (topic_id);

-- Starting list. Unlike everything else here this is *data*, kept in the same file so a
-- fresh database is usable without a manual step; the admin owns it from then on, and it
-- must stay identical to the list documented in AGENTS.md section 11.
INSERT INTO topics (key, name) VALUES
    ('ai', 'هوش مصنوعی'),
    ('startup', 'استارتاپ')
ON CONFLICT (key) DO NOTHING;

INSERT INTO sources (topic_id, rss_url)
SELECT seed_topic.id, seed.rss_url
FROM (
    VALUES
        ('ai', 'https://www.reddit.com/r/mlops/new/.rss'),
        ('startup', 'https://www.reddit.com/r/venturecapital/new/.rss')
) AS seed(topic_key, rss_url)
JOIN topics AS seed_topic ON seed_topic.key = seed.topic_key
ON CONFLICT (rss_url) DO NOTHING;

CREATE TABLE IF NOT EXISTS posts (
    id                  BIGSERIAL PRIMARY KEY,
    reddit_id           TEXT NOT NULL UNIQUE,        -- RSS feed guid, e.g. t3_1abcde
    subreddit           TEXT NOT NULL,
    source_topic_key    TEXT NOT NULL,               -- topic key at fetch time (not a FK: audit survives a delete)
    title               TEXT NOT NULL,
    url                 TEXT NOT NULL,
    author              TEXT,
    raw_content         TEXT,                        -- raw text/summary from RSS
    posted_at           TIMESTAMPTZ,                 -- when Reddit published it (RSS timestamp)
    fetched_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- machine state of the post through the pipeline (AGENTS.md section 10)
    status              TEXT NOT NULL DEFAULT 'new'
                        CHECK (status IN (
                            'new', 'awaiting_review', 'approved', 'rejected',
                            'skipped_irrelevant', 'skipped_duplicate',
                            'skipped_low_importance', 'to_send', 'publishing',
                            'sent', 'failed'
                        )),
    -- the human decision, recorded separately from the machine state (phase 5)
    review_status       TEXT NOT NULL DEFAULT 'pending_review'
                        CHECK (review_status IN ('pending_review', 'approved', 'rejected')),

    -- review audit trail (FR-12)
    reviewed_by         TEXT,                        -- Telegram user id of the deciding admin
    reviewed_by_name    TEXT,                        -- that admin's display name (phase 6)
    reviewed_at         TIMESTAMPTZ,
    approved_at         TIMESTAMPTZ,
    rejected_at         TIMESTAMPTZ,
    private_channel_id  TEXT,                        -- review channel + message that carries the buttons
    private_message_id  BIGINT,

    -- LLM analysis output, filled only after the admin approved (FR-13)
    is_relevant         BOOLEAN,
    duplicate_of_id     BIGINT REFERENCES posts(id),
    topic               TEXT,                        -- LLM-confirmed topic (from the allowed list)
    importance          TEXT CHECK (importance IN ('low','medium','high')),
    summary_fa          TEXT,
    key_points          JSONB,                       -- ["نکته ۱", "نکته ۲", ...]
    llm_raw_response    JSONB,                       -- raw LLM output for audit/debug
    ai_processed_at     TIMESTAMPTZ,
    ai_error            TEXT,                        -- why the AI step failed, if it did

    -- publication
    public_channel_id   TEXT,
    public_message_id   BIGINT,
    published_at        TIMESTAMPTZ,                 -- when the public channel got the message

    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_posts_status         ON posts (status);
CREATE INDEX IF NOT EXISTS idx_posts_review_status  ON posts (review_status);
CREATE INDEX IF NOT EXISTS idx_posts_posted_at      ON posts (posted_at DESC);
