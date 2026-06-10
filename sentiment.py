"""
sentiment.py — Phase 4: Sentiment Analysis Layer
=================================================
Uses Hugging Face transformers pipeline with the
cardiffnlp/twitter-roberta-base-sentiment model.

The model outputs three labels:
  LABEL_0 → Negative
  LABEL_1 → Neutral
  LABEL_2 → Positive

Main public functions
---------------------
analyze_comments(channel_id, batch_size)  → SentimentResult
analyze_texts(texts, batch_size)          → SentimentResult

SentimentResult (dict)
-----------------------
{
  "total":    int,
  "positive": {"count": int, "pct": float},
  "neutral":  {"count": int, "pct": float},
  "negative": {"count": int, "pct": float},
  "samples": {
      "positive": list[str],   # up to 3 example comments
      "neutral":  list[str],
      "negative": list[str],
  }
}
"""
from __future__ import annotations

import warnings
from typing import Optional

warnings.filterwarnings("ignore")

import db

# ── Model ID ──────────────────────────────────────────────────────────────────
_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment"

# Label mapping for this model
_LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    # Some versions of the model use readable labels directly
    "Negative": "negative",
    "Neutral":  "neutral",
    "Positive": "positive",
}

# ── Lazy pipeline loader ───────────────────────────────────────────────────────
_pipeline = None

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        from rich.console import Console
        Console().print(
            "[cyan]Loading sentiment model "
            f"[bold]{_MODEL_ID}[/bold] "
            "(first run downloads ~500 MB)…[/cyan]"
        )
        _pipeline = pipeline(
            "sentiment-analysis",
            model=_MODEL_ID,
            tokenizer=_MODEL_ID,
            truncation=True,
            max_length=128,
        )
    return _pipeline


# ── Text preprocessing ─────────────────────────────────────────────────────────

def _preprocess(text: str) -> str:
    """Basic cleanup: strip excess whitespace, cap length."""
    import re
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:512]


# ── Core analysis ──────────────────────────────────────────────────────────────

def analyze_texts(
    texts: list[str],
    batch_size: int = 32,
) -> dict:
    """
    Run sentiment analysis on a list of strings.
    Returns a SentimentResult dict.
    """
    import re
    if not texts:
        return _empty_result()

    try:
        pipe = _get_pipeline()
        use_pipeline = True
    except Exception as e:
        use_pipeline = False
        from rich.console import Console
        Console().print(
            f"[yellow]Sentiment model could not be loaded ({e}). "
            "Using fast lexicon fallback...[/yellow]"
        )

    cleaned = [_preprocess(t) for t in texts if t and t.strip()]
    if not cleaned:
        return _empty_result()

    cleaned = cleaned[:1000]

    counts = {"positive": 0, "neutral": 0, "negative": 0}
    samples = {"positive": [], "neutral": [], "negative": []}

    if use_pipeline:
        results = []
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i : i + batch_size]
            try:
                preds = pipe(batch, batch_size=batch_size)
                results.extend(preds)
            except Exception as e:
                # Skip bad batches
                results.extend([{"label": "LABEL_1", "score": 1.0}] * len(batch))

        for text, pred in zip(cleaned, results):
            raw_label = pred.get("label", "LABEL_1")
            label     = _LABEL_MAP.get(raw_label, "neutral")
            counts[label] += 1
            if len(samples[label]) < 3:
                samples[label].append(text[:120])
    else:
        # Lexicon fallback
        pos_words = {
            "great", "good", "love", "best", "awesome", "perfect", "amazing", "cool", "helpful", "thanks", "nice",
            "excellent", "superb", "brilliant", "fantastic", "interesting", "useful", "like", "appreciate", "glad",
            "wonderful", "recommend", "smart", "easy", "clear", "incredible", "favorite", "genius", "wow"
        }
        neg_words = {
            "bad", "worst", "fail", "terrible", "boring", "suck", "hate", "wrong", "useless", "dislike", "horrible",
            "waste", "annoying", "poor", "difficult", "stupid", "hard", "confusing", "slow", "broken", "sad"
        }
        for text in cleaned:
            words = re.findall(r'\b\w+\b', text.lower())
            pos_count = sum(1 for w in words if w in pos_words)
            neg_count = sum(1 for w in words if w in neg_words)
            if pos_count > neg_count:
                label = "positive"
            elif neg_count > pos_count:
                label = "negative"
            else:
                label = "neutral"
            counts[label] += 1
            if len(samples[label]) < 3:
                samples[label].append(text[:120])

    total = sum(counts.values())

    def pct(n):
        return round(n / total * 100, 1) if total else 0.0

    return {
        "total":    total,
        "positive": {"count": counts["positive"], "pct": pct(counts["positive"])},
        "neutral":  {"count": counts["neutral"],  "pct": pct(counts["neutral"])},
        "negative": {"count": counts["negative"], "pct": pct(counts["negative"])},
        "samples":  samples,
    }


def _empty_result() -> dict:
    return {
        "total":    0,
        "positive": {"count": 0, "pct": 0.0},
        "neutral":  {"count": 0, "pct": 0.0},
        "negative": {"count": 0, "pct": 0.0},
        "samples":  {"positive": [], "neutral": [], "negative": []},
    }


# ── DB-integrated helpers ──────────────────────────────────────────────────────

def analyze_comments(
    channel_id: Optional[str] = None,
    batch_size: int = 32,
) -> dict:
    """
    Pull comments from DB and run sentiment analysis.
    Optionally scoped to a single channel.
    """
    with db.get_conn() as conn:
        if channel_id:
            rows = conn.execute("""
                SELECT tc.text FROM top_comments tc
                JOIN videos v ON tc.video_id = v.video_id
                WHERE v.channel_id = ?
                  AND length(tc.text) > 5
            """, (channel_id,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT text FROM top_comments
                WHERE length(text) > 5
            """).fetchall()

    texts = [r["text"] for r in rows]
    return analyze_texts(texts, batch_size=batch_size)


def analyze_titles(
    channel_id: Optional[str] = None,
    batch_size: int = 32,
) -> dict:
    """Run sentiment analysis on video titles."""
    if channel_id:
        videos = db.get_videos_for_channel(channel_id)
    else:
        videos = db.get_all_videos()

    texts = [v["title"] for v in videos if v["title"]]
    return analyze_texts(texts, batch_size=batch_size)
