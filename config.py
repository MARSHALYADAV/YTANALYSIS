"""
config.py — Central configuration for YouTube Analytics System.
Loads .env and exposes all constants used across modules.

Phases covered
--------------
  1-2  YouTube Data API v3, SQLite
  3    NLP (sentence-transformers, scikit-learn)
  4    Sentiment (Hugging Face)
  5    Gemini Insight Layer (Google AI Studio free tier)
  6    Recommendation Engine
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# ── YouTube API ───────────────────────────────────────────────────────────────
YOUTUBE_API_KEY: str = os.getenv("YOUTUBE_API_KEY", "")
YOUTUBE_API_SERVICE = "youtube"
YOUTUBE_API_VERSION = "v3"

if not YOUTUBE_API_KEY or YOUTUBE_API_KEY == "YOUR_API_KEY_HERE":
    import sys
    print(
        "[ERROR] No YouTube API key found.\n"
        "  1. Copy .env.example → .env\n"
        "  2. Set YOUTUBE_API_KEY=your_key_here\n"
        "  Get a key at: https://console.cloud.google.com"
    )
    sys.exit(1)

# ── Gemini AI API (Phase 5) ───────────────────────────────────────────────────
# Free tier from Google AI Studio: https://aistudio.google.com/apikey
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ── Database ──────────────────────────────────────────────────────────────────
_db_path_str: str = os.getenv("DB_PATH", "data/youtube.db")
DB_PATH: Path = BASE_DIR / _db_path_str
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Reports output dir ─────────────────────────────────────────────────────────
REPORTS_DIR: Path = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Collection defaults ────────────────────────────────────────────────────────
DEFAULT_MAX_PAGES     = 5     # video list pages (50 videos each)
DEFAULT_TOP_COMMENTS  = 20    # top comments fetched per video
COMMENTS_BATCH_SIZE   = 50    # YouTube API max per request

# ── Trend Score weights ────────────────────────────────────────────────────────
TREND_WEIGHT_VIEWS    = 0.4
TREND_WEIGHT_COMMENTS = 0.2
TREND_WEIGHT_LIKES    = 0.2
TREND_WEIGHT_RECENCY  = 0.2
TREND_RECENCY_HALFLIFE_DAYS = 30  # exponential decay half-life

# ── Creator Intelligence Score weights ────────────────────────────────────────
CREATOR_WEIGHT_ENGAGEMENT   = 0.4
CREATOR_WEIGHT_CONSISTENCY  = 0.3
CREATOR_WEIGHT_GROWTH       = 0.3

# ── Recommendation Engine weights (Phase 6) ───────────────────────────────────
RECO_WEIGHT_TREND       = 0.40   # topic trend score
RECO_WEIGHT_SENTIMENT   = 0.35   # positive sentiment %
RECO_WEIGHT_ENGAGEMENT  = 0.25   # channel engagement rate

# ── Score category labels ─────────────────────────────────────────────────────
SCORE_CATEGORIES = [
    (85, "🔥 Elite Creator"),
    (70, "🚀 High Potential"),
    (50, "📈 Growing"),
    (0,  "🌱 Early Stage"),
]

def score_label(score: float) -> str:
    """Return category label for a 0–100 score."""
    for threshold, label in SCORE_CATEGORIES:
        if score >= threshold:
            return label
    return "🌱 Early Stage"
