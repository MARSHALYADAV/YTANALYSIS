"""
nlp.py — Phase 3: NLP Topic Extraction Layer
=============================================
Uses sentence-transformers (all-MiniLM-L6-v2) to embed video titles
and comment text, then clusters with KMeans (scikit-learn) to surface
latent topics.  Also extracts per-topic keyword labels via TF-IDF.

Main public functions
---------------------
extract_topics_from_titles(channel_id, n_topics)  → list[Topic]
extract_topics_from_comments(channel_id, n_topics) → list[Topic]
extract_keywords_from_text(texts, top_n)           → list[str]

Topic dataclass
---------------
{
  "id":         int,
  "label":      str,          # e.g. "AI Agents"
  "keywords":   list[str],    # top keywords in this cluster
  "size":       int,          # number of texts in this cluster
  "examples":   list[str],    # up to 3 representative texts
}
"""
from __future__ import annotations

import re
import json
import warnings
from collections import Counter
from typing import Optional

warnings.filterwarnings("ignore")   # suppress transformers / sklearn warnings

# ── Lazy imports (heavy models load only when needed) ─────────────────────────
_model = None
_vectorizer_cls = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from rich.console import Console
        Console().print("[cyan]Loading sentence-transformer model (first run downloads ~80 MB)…[/cyan]")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


import db

# ── Stopwords (reuse the same set as analyzer.py) ─────────────────────────────
_STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "this","that","is","are","was","were","be","been","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall",
    "not","no","nor","so","yet","both","either","neither","each","few","more",
    "most","other","some","such","than","then","these","those","up","out",
    "if","its","it","my","me","we","us","our","you","your","he","she","they",
    "them","their","what","which","who","how","when","where","why","video",
    "new","get","more","watch","official","full","best","top","all","one",
    "like","just","know","want","think","really","good","time","make","see",
    "go","going","also","back","after","use","two","how","our","work","well",
    "even","way","because","come","its","here","made","too","say","great",
    "much","very","still","own","before","between","need","never","every",
    "last","many","never","always","still","own","since","without","while",
}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


# ── Core clustering ────────────────────────────────────────────────────────────

def _cluster_texts(
    texts: list[str],
    n_topics: int,
) -> list[dict]:
    """
    Embed texts with sentence-transformers, cluster with KMeans.
    Returns a list of topic dicts.
    """
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    if len(texts) < n_topics:
        n_topics = max(2, len(texts) // 2)

    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False)

    km = KMeans(n_clusters=n_topics, random_state=42, n_init="auto")
    labels = km.fit_predict(embeddings)

    # Group texts by cluster
    clusters: dict[int, list[str]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(texts[idx])

    # TF-IDF keywords per cluster
    topics = []
    for cluster_id, cluster_texts in sorted(clusters.items(), key=lambda x: -len(x[1])):
        joined = " ".join(cluster_texts)
        tokens = _tokenize(joined)
        token_freq = Counter(tokens)

        # Top keywords for this cluster
        keywords = [w for w, _ in token_freq.most_common(6)]

        # Auto-label: capitalize top 2 keywords
        label_words = [w.capitalize() for w in keywords[:2]]
        label = " ".join(label_words) if label_words else f"Topic {cluster_id + 1}"

        topics.append({
            "id":       cluster_id,
            "label":    label,
            "keywords": keywords,
            "size":     len(cluster_texts),
            "examples": cluster_texts[:3],
        })

    # Sort by size descending
    return sorted(topics, key=lambda t: -t["size"])


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_topics_from_titles(
    channel_id: Optional[str] = None,
    n_topics: int = 6,
) -> list[dict]:
    """
    Extract topics from video titles (optionally scoped to one channel).
    """
    if channel_id:
        videos = db.get_videos_for_channel(channel_id)
    else:
        videos = db.get_all_videos()

    titles = [v["title"] for v in videos if v["title"] and len(v["title"]) > 5]
    if not titles:
        return []

    return _cluster_texts(titles, n_topics)


def extract_topics_from_comments(
    channel_id: Optional[str] = None,
    n_topics: int = 5,
) -> list[dict]:
    """
    Extract topics from top comments for a channel's videos.
    """
    with db.get_conn() as conn:
        if channel_id:
            rows = conn.execute("""
                SELECT tc.text FROM top_comments tc
                JOIN videos v ON tc.video_id = v.video_id
                WHERE v.channel_id = ?
                  AND length(tc.text) > 10
            """, (channel_id,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT text FROM top_comments
                WHERE length(text) > 10
            """).fetchall()

    texts = [r["text"] for r in rows]
    if not texts:
        return []

    # Cap at 2000 comments for speed
    texts = texts[:2000]
    return _cluster_texts(texts, n_topics)


def extract_keywords_from_text(
    texts: list[str],
    top_n: int = 10,
) -> list[str]:
    """
    Quick TF-IDF keyword extraction from a flat list of strings.
    Returns a list of top_n keywords.
    """
    if not texts:
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    def tokenizer(text):
        return _tokenize(text)

    vec = TfidfVectorizer(tokenizer=tokenizer, max_features=500, token_pattern=None)
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return []

    scores = np.asarray(X.mean(axis=0)).flatten()
    vocab  = vec.get_feature_names_out()
    ranked = sorted(zip(vocab, scores), key=lambda x: -x[1])
    return [w for w, _ in ranked[:top_n]]
