"""
recommender.py — Phase 6: Content Recommendation Engine
========================================================
Scores extracted NLP topics using a weighted combination of:
  • Trend Score     (how trending videos in this topic cluster are)
  • Positive Sentiment % (audience enthusiasm from Phase 4)
  • Engagement Rate  (creator's base engagement from Phase 2)

Formula
-------
recommendation_score =
    trend_score          × RECO_WEIGHT_TREND      (0.40)  +
    positive_sentiment   × RECO_WEIGHT_SENTIMENT  (0.35)  +
    engagement_rate      × RECO_WEIGHT_ENGAGEMENT (0.25)

All three inputs are normalised to [0, 100] before weighting.

Public API
----------
recommend_topics(
    channel_id,
    topics,           # from nlp.extract_topics_from_titles()
    sentiment_result, # from sentiment.analyze_comments()
    top_n,
) → list[RecommendationResult]

RecommendationResult dict
--------------------------
{
  "rank":            int,
  "topic":           str,       # "AI Agents"
  "keywords":        list[str],
  "score":           float,     # 0–100
  "trend_component": float,
  "sent_component":  float,
  "eng_component":   float,
  "size":            int,       # number of videos/comments in cluster
  "examples":        list[str], # up to 3 representative titles
}
"""
from __future__ import annotations

from typing import Optional

import config
import analyzer
import db


# ── Score normalisation ────────────────────────────────────────────────────────

def _normalize_list(values: list[float]) -> list[float]:
    """Min-max normalise to [0, 100]. Returns 50s if all identical."""
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mx == mn:
        return [50.0] * len(values)
    return [(v - mn) / (mx - mn) * 100 for v in values]


# ── Per-topic trend score ──────────────────────────────────────────────────────

def _topic_trend_score(topic: dict, all_videos_scored: list[dict]) -> float:
    """
    Match videos whose titles appear in this topic's examples/keywords
    and average their trend scores. Falls back to 50 if no match.
    """
    keywords_lower = {k.lower() for k in topic.get("keywords", [])}
    examples_lower = {e.lower() for e in topic.get("examples", [])}

    matched_scores = []
    for v in all_videos_scored:
        title_lower = v.get("title", "").lower()
        # Match if any keyword appears in the video title
        if any(kw in title_lower for kw in keywords_lower):
            matched_scores.append(v["trend_score"])
        # Or if the title is one of the cluster examples
        elif any(ex[:40].lower() in title_lower for ex in examples_lower):
            matched_scores.append(v["trend_score"])

    if not matched_scores:
        return 50.0  # neutral default
    return sum(matched_scores) / len(matched_scores)


# ── Main recommendation function ───────────────────────────────────────────────

def recommend_topics(
    channel_id: str,
    topics: list[dict],
    sentiment_result: Optional[dict] = None,
    top_n: int = 10,
) -> list[dict]:
    """
    Score and rank NLP topic clusters for content recommendation.

    Parameters
    ----------
    channel_id       : channel in DB (used for engagement rate)
    topics           : output of nlp.extract_topics_from_titles()
    sentiment_result : output of sentiment.analyze_comments()
    top_n            : how many recommendations to return

    Returns
    -------
    List of RecommendationResult dicts, sorted by score descending.
    """
    if not topics:
        return []

    # ── Component 1: Trend Score per topic ───────────────────────────────
    all_videos_scored = analyzer.compute_trend_scores(
        db.get_videos_for_channel(channel_id)
    )
    raw_trend = [_topic_trend_score(t, all_videos_scored) for t in topics]

    # ── Component 2: Positive Sentiment (uniform across topics) ──────────
    pos_pct = 50.0  # default if no sentiment data
    if sentiment_result and sentiment_result.get("total", 0) > 0:
        pos_pct = sentiment_result["positive"]["pct"]

    # ── Component 3: Engagement Rate ─────────────────────────────────────
    creator = analyzer.compute_creator_score(channel_id)
    eng_rate = creator.get("engagement_rate", 50.0)

    # ── Normalise trend scores ─────────────────────────────────────────────
    norm_trend = _normalize_list(raw_trend)

    # ── Compute recommendation score ──────────────────────────────────────
    results = []
    for i, (topic, norm_t, raw_t) in enumerate(zip(topics, norm_trend, raw_trend)):
        score = (
            norm_t   * config.RECO_WEIGHT_TREND      +
            pos_pct  * config.RECO_WEIGHT_SENTIMENT  +
            eng_rate * config.RECO_WEIGHT_ENGAGEMENT
        )
        score = round(min(score, 100.0), 1)

        results.append({
            "rank":            0,      # filled below
            "topic":           topic["label"],
            "keywords":        topic.get("keywords", []),
            "score":           score,
            "trend_component": round(norm_t,   1),
            "sent_component":  round(pos_pct,  1),
            "eng_component":   round(eng_rate, 1),
            "raw_trend":       round(raw_t,    1),
            "size":            topic.get("size", 0),
            "examples":        topic.get("examples", [])[:3],
        })

    # ── Sort and rank ─────────────────────────────────────────────────────
    results.sort(key=lambda x: -x["score"])
    for rank, r in enumerate(results[:top_n], 1):
        r["rank"] = rank

    return results[:top_n]


# ── Standalone quick score (no NLP needed) ─────────────────────────────────────

def quick_recommend_from_keywords(
    channel_id: str,
    keywords: list[tuple[str, float]],      # from analyzer.trending_keywords()
    sentiment_result: Optional[dict] = None,
    top_n: int = 10,
) -> list[dict]:
    """
    Lightweight recommendation using TF-IDF keywords (no NLP clustering needed).
    Each keyword becomes its own "topic".

    Parameters
    ----------
    keywords : list of (word, tfidf_score) from analyzer.trending_keywords()
    """
    if not keywords:
        return []

    creator  = analyzer.compute_creator_score(channel_id)
    eng_rate = creator.get("engagement_rate", 50.0)

    pos_pct = 50.0
    if sentiment_result and sentiment_result.get("total", 0) > 0:
        pos_pct = sentiment_result["positive"]["pct"]

    # Normalise TF-IDF scores to 0-100
    raw_scores = [s for _, s in keywords]
    mn, mx = min(raw_scores), max(raw_scores)
    span   = mx - mn or 1

    results = []
    for word, raw_s in keywords[:top_n]:
        norm_t = (raw_s - mn) / span * 100
        score  = (
            norm_t   * config.RECO_WEIGHT_TREND      +
            pos_pct  * config.RECO_WEIGHT_SENTIMENT  +
            eng_rate * config.RECO_WEIGHT_ENGAGEMENT
        )
        results.append({
            "rank":            0,
            "topic":           word.capitalize(),
            "keywords":        [word],
            "score":           round(min(score, 100.0), 1),
            "trend_component": round(norm_t,   1),
            "sent_component":  round(pos_pct,  1),
            "eng_component":   round(eng_rate, 1),
            "size":            0,
            "examples":        [],
        })

    results.sort(key=lambda x: -x["score"])
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    return results
