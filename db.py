"""
db.py — SQLite schema and upsert helpers for YouTube Analytics System.

Tables
------
channels      — channel-level metadata + stats
videos        — per-video stats
top_comments  — top comments per video
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from config import DB_PATH

# ── Schema DDL ────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS channels (
    channel_id       TEXT PRIMARY KEY,
    title            TEXT,
    description      TEXT,
    custom_url       TEXT,
    country          TEXT,
    subscriber_count INTEGER,
    view_count       INTEGER,
    video_count      INTEGER,
    published_at     TEXT,
    thumbnail_url    TEXT,
    fetched_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS videos (
    video_id         TEXT PRIMARY KEY,
    channel_id       TEXT NOT NULL REFERENCES channels(channel_id),
    title            TEXT,
    description      TEXT,
    published_at     TEXT,
    duration         TEXT,
    view_count       INTEGER DEFAULT 0,
    like_count       INTEGER DEFAULT 0,
    comment_count    INTEGER DEFAULT 0,
    thumbnail_url    TEXT,
    tags             TEXT,        -- JSON array as TEXT
    fetched_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS top_comments (
    comment_id       TEXT PRIMARY KEY,
    video_id         TEXT NOT NULL REFERENCES videos(video_id),
    author           TEXT,
    text             TEXT,
    like_count       INTEGER DEFAULT 0,
    published_at     TEXT,
    fetched_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_videos_channel   ON videos(channel_id);
CREATE INDEX IF NOT EXISTS idx_videos_published ON videos(published_at);
CREATE INDEX IF NOT EXISTS idx_comments_video   ON top_comments(video_id);
"""

# ── Connection context manager ─────────────────────────────────────────────────

@contextmanager
def get_conn() -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection with row_factory and auto-commit on success."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript(_DDL)


# ── Upsert helpers ─────────────────────────────────────────────────────────────

def upsert_channel(data: dict) -> None:
    sql = """
    INSERT INTO channels
        (channel_id, title, description, custom_url, country,
         subscriber_count, view_count, video_count, published_at,
         thumbnail_url, fetched_at)
    VALUES
        (:channel_id, :title, :description, :custom_url, :country,
         :subscriber_count, :view_count, :video_count, :published_at,
         :thumbnail_url, datetime('now'))
    ON CONFLICT(channel_id) DO UPDATE SET
        title            = excluded.title,
        description      = excluded.description,
        custom_url       = excluded.custom_url,
        country          = excluded.country,
        subscriber_count = excluded.subscriber_count,
        view_count       = excluded.view_count,
        video_count      = excluded.video_count,
        published_at     = excluded.published_at,
        thumbnail_url    = excluded.thumbnail_url,
        fetched_at       = excluded.fetched_at
    """
    with get_conn() as conn:
        conn.execute(sql, data)


def upsert_videos(rows: list[dict]) -> None:
    sql = """
    INSERT INTO videos
        (video_id, channel_id, title, description, published_at, duration,
         view_count, like_count, comment_count, thumbnail_url, tags, fetched_at)
    VALUES
        (:video_id, :channel_id, :title, :description, :published_at, :duration,
         :view_count, :like_count, :comment_count, :thumbnail_url, :tags, datetime('now'))
    ON CONFLICT(video_id) DO UPDATE SET
        title         = excluded.title,
        description   = excluded.description,
        published_at  = excluded.published_at,
        duration      = excluded.duration,
        view_count    = excluded.view_count,
        like_count    = excluded.like_count,
        comment_count = excluded.comment_count,
        thumbnail_url = excluded.thumbnail_url,
        tags          = excluded.tags,
        fetched_at    = excluded.fetched_at
    """
    with get_conn() as conn:
        conn.executemany(sql, rows)


def upsert_comments(rows: list[dict]) -> None:
    sql = """
    INSERT INTO top_comments
        (comment_id, video_id, author, text, like_count, published_at, fetched_at)
    VALUES
        (:comment_id, :video_id, :author, :text, :like_count, :published_at, datetime('now'))
    ON CONFLICT(comment_id) DO UPDATE SET
        like_count   = excluded.like_count,
        fetched_at   = excluded.fetched_at
    """
    with get_conn() as conn:
        conn.executemany(sql, rows)


# ── Query helpers ──────────────────────────────────────────────────────────────

def get_channel(channel_id: str) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM channels WHERE channel_id=?", (channel_id,)
        ).fetchone()


def get_all_channels() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM channels ORDER BY subscriber_count DESC"
        ).fetchall()


def get_videos_for_channel(channel_id: str) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM videos WHERE channel_id=? ORDER BY view_count DESC",
            (channel_id,)
        ).fetchall()


def get_all_videos() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM videos ORDER BY view_count DESC"
        ).fetchall()


def get_comments_for_video(video_id: str) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM top_comments WHERE video_id=? ORDER BY like_count DESC",
            (video_id,)
        ).fetchall()
