"""
analyzer.py — Analysis layer for YouTube Analytics System.

Computes:
  • Trend Score        (per video)
  • Creator Intelligence Score  (per channel)
  • Fastest-growing videos / channels
  • Trending keywords  (TF-IDF on titles + tags)
"""
import json
import math
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

import db
import config

# ── Date helpers ───────────────────────────────────────────────────────────────

def _parse_iso(dt_str: str) -> Optional[datetime]:
    """Parse an ISO-8601 string to an aware datetime (UTC)."""
    if not dt_str:
        return None
    try:
        dt_str = dt_str.rstrip("Z")
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _days_ago(dt: datetime) -> float:
    """Return how many days ago dt was (from now UTC)."""
    now = datetime.now(timezone.utc)
    delta = now - dt
    return max(delta.total_seconds() / 86400, 0)


# ── Recency score ──────────────────────────────────────────────────────────────

def recency_score(published_at: str, half_life_days: float = config.TREND_RECENCY_HALFLIFE_DAYS) -> float:
    """
    Exponential decay: score = 100 × 2^(−age/half_life)
    A video published today  → 100
    A video published 30d ago → 50
    A video published 90d ago → ~12.5
    """
    dt = _parse_iso(published_at)
    if not dt:
        return 0.0
    age = _days_ago(dt)
    return 100.0 * math.pow(2, -age / half_life_days)


# ── Normalisation helper ───────────────────────────────────────────────────────

def _normalize(values: list[float]) -> list[float]:
    """Min-max normalize a list to [0, 100]. Returns zeros if all same."""
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mx == mn:
        return [50.0] * len(values)
    return [(v - mn) / (mx - mn) * 100 for v in values]


# ── Trend Score ────────────────────────────────────────────────────────────────

def compute_trend_scores(videos: list[sqlite3.Row]) -> list[dict]:
    """
    For a list of video rows, compute and return enriched dicts with trend_score.

    trend_score =
        norm(views)    × TREND_WEIGHT_VIEWS    +
        norm(comments) × TREND_WEIGHT_COMMENTS +
        norm(likes)    × TREND_WEIGHT_LIKES    +
        recency        × TREND_WEIGHT_RECENCY
    """
    if not videos:
        return []

    raw_views    = [int(v["view_count"])    for v in videos]
    raw_comments = [int(v["comment_count"]) for v in videos]
    raw_likes    = [int(v["like_count"])    for v in videos]
    raw_recency  = [recency_score(v["published_at"]) for v in videos]

    norm_v = _normalize(raw_views)
    norm_c = _normalize(raw_comments)
    norm_l = _normalize(raw_likes)
    # recency is already 0–100

    results = []
    for i, v in enumerate(videos):
        score = (
            norm_v[i]        * config.TREND_WEIGHT_VIEWS    +
            norm_c[i]        * config.TREND_WEIGHT_COMMENTS +
            norm_l[i]        * config.TREND_WEIGHT_LIKES    +
            raw_recency[i]   * config.TREND_WEIGHT_RECENCY
        )
        results.append({
            "video_id":     v["video_id"],
            "channel_id":   v["channel_id"],
            "title":        v["title"],
            "published_at": v["published_at"],
            "view_count":   int(v["view_count"]),
            "like_count":   int(v["like_count"]),
            "comment_count":int(v["comment_count"]),
            "recency_score":round(raw_recency[i], 1),
            "trend_score":  round(min(score, 100.0), 1),
        })

    return sorted(results, key=lambda x: x["trend_score"], reverse=True)


# ── Creator Intelligence Score ─────────────────────────────────────────────────

def _engagement_rate(videos: list[sqlite3.Row]) -> float:
    """Average (likes + comments) / views across all videos. Returns 0–100."""
    if not videos:
        return 0.0
    rates = []
    for v in videos:
        views = int(v["view_count"]) or 1
        eng   = (int(v["like_count"]) + int(v["comment_count"])) / views
        rates.append(eng)
    avg = sum(rates) / len(rates)
    # Cap at 10% engagement = 100 score (most creators are <5%)
    return min(avg / 0.10 * 100, 100.0)


def _upload_consistency(videos: list[sqlite3.Row]) -> float:
    """
    Lower std-dev of gaps between uploads → higher consistency score (0–100).
    Needs ≥2 videos. Returns 50 if < 2 videos.
    """
    if len(videos) < 2:
        return 50.0

    dates = sorted(
        [_parse_iso(v["published_at"]) for v in videos if _parse_iso(v["published_at"])]
    )
    if len(dates) < 2:
        return 50.0

    gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
    mean = sum(gaps) / len(gaps)
    variance = sum((g - mean) ** 2 for g in gaps) / len(gaps)
    std_dev   = math.sqrt(variance)

    # std_dev of 0 → 100, std_dev of 60 days → 0
    score = max(0.0, 100.0 - (std_dev / 60.0) * 100)
    return round(score, 1)


def _growth_rate_score(channel: sqlite3.Row, videos: list[sqlite3.Row]) -> float:
    """
    Proxy growth rate: views-per-day of recent videos vs older videos.
    Recent = last 25% of videos by publish date. Score 0–100.
    """
    if not videos:
        return 0.0

    sorted_vids = sorted(
        [v for v in videos if _parse_iso(v["published_at"])],
        key=lambda v: v["published_at"]
    )
    n = len(sorted_vids)
    if n < 4:
        return 50.0

    split      = n * 3 // 4
    older      = sorted_vids[:split]
    recent     = sorted_vids[split:]

    def vpd(vids):
        """Average views-per-day."""
        rates = []
        for v in vids:
            dt = _parse_iso(v["published_at"])
            age = _days_ago(dt) or 1
            rates.append(int(v["view_count"]) / age)
        return sum(rates) / len(rates) if rates else 0

    old_vpd    = vpd(older)
    recent_vpd = vpd(recent)

    if old_vpd == 0:
        return 50.0

    ratio = recent_vpd / old_vpd  # > 1 means growing
    # ratio of 2× → 100, 1× → 50, 0.5× → 0
    score = (ratio - 0.5) / 1.5 * 100
    return round(max(0.0, min(score, 100.0)), 1)


def compute_creator_score(channel_id: str) -> dict:
    """
    Returns a dict with creator_score (0–100) and component scores.

    creator_score =
        engagement_rate      × CREATOR_WEIGHT_ENGAGEMENT  +
        upload_consistency   × CREATOR_WEIGHT_CONSISTENCY +
        growth_rate          × CREATOR_WEIGHT_GROWTH
    """
    channel = db.get_channel(channel_id)
    videos  = db.get_videos_for_channel(channel_id)

    if not channel:
        return {"error": f"Channel {channel_id} not in database. Run collect first."}

    eng  = _engagement_rate(videos)
    cons = _upload_consistency(videos)
    grow = _growth_rate_score(channel, videos)

    score = (
        eng  * config.CREATOR_WEIGHT_ENGAGEMENT  +
        cons * config.CREATOR_WEIGHT_CONSISTENCY +
        grow * config.CREATOR_WEIGHT_GROWTH
    )
    score = round(min(score, 100.0), 1)
    label = config.score_label(score)

    return {
        "channel_id":         channel_id,
        "channel_title":      channel["title"],
        "subscriber_count":   channel["subscriber_count"],
        "video_count":        len(videos),
        "engagement_rate":    round(eng,  1),
        "upload_consistency": round(cons, 1),
        "growth_rate":        round(grow, 1),
        "creator_score":      score,
        "label":              label,
    }


# ── Fastest-growing videos ─────────────────────────────────────────────────────

def fastest_growing_videos(top_n: int = 10) -> list[dict]:
    """Return top N videos by trend score across all channels in the DB."""
    videos = db.get_all_videos()
    scored = compute_trend_scores(videos)
    return scored[:top_n]


# ── Fastest-growing channels ───────────────────────────────────────────────────

def fastest_growing_channels(top_n: int = 10) -> list[dict]:
    """
    Return top N channels by growth_rate component of creator score.
    """
    channels = db.get_all_channels()
    results  = []
    for ch in channels:
        result = compute_creator_score(ch["channel_id"])
        if "error" not in result:
            results.append(result)
    return sorted(results, key=lambda x: x["growth_rate"], reverse=True)[:top_n]


# ── Trending keywords (TF-IDF) ─────────────────────────────────────────────────

_STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "this","that","is","are","was","were","be","been","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall",
    "not","no","nor","so","yet","both","either","neither","each","few","more",
    "most","other","some","such","than","then","these","those","up","out",
    "if","its","it","my","me","we","us","our","you","your","he","she","they",
    "them","their","what","which","who","how","when","where","why","video",
    "new","get","more","watch","official","full","best","top","all","one",
}

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [w for w in words if w not in _STOPWORDS]


def trending_keywords(top_n: int = 20) -> list[tuple[str, int]]:
    """
    TF-IDF on video titles + tags across all videos.
    Returns list of (keyword, score) sorted descending.
    """
    videos = db.get_all_videos()
    if not videos:
        return []

    # Build corpus: each "document" = title + tags
    corpus: list[list[str]] = []
    for v in videos:
        tags  = json.loads(v["tags"] or "[]") if v["tags"] else []
        text  = v["title"] + " " + " ".join(tags)
        tokens = _tokenize(text)
        corpus.append(tokens)

    N = len(corpus)  # total documents

    # Term frequency per document
    tf_per_doc = [Counter(doc) for doc in corpus]

    # Document frequency (how many docs contain the term)
    df: Counter = Counter()
    for doc in corpus:
        df.update(set(doc))

    # TF-IDF: sum across all docs
    global_tfidf: Counter = Counter()
    for tf in tf_per_doc:
        for term, count in tf.items():
            idf = math.log((N + 1) / (df[term] + 1)) + 1
            global_tfidf[term] += count * idf

    return global_tfidf.most_common(top_n)
