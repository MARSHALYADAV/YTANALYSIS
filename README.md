# 📺 YouTube Analytics System

> A full-stack YouTube intelligence tool built with Python — featuring 6 modular phases from data collection to AI-powered insights.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![YouTube API](https://img.shields.io/badge/YouTube-Data%20API%20v3-red?logo=youtube)](https://developers.google.com/youtube/v3)
[![Gemini](https://img.shields.io/badge/Gemini-AI%20Studio-orange?logo=google)](https://aistudio.google.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🏗️ Architecture — 6 Phases

```
Phase 1 │ Data Collection     → YouTube API v3 → SQLite
Phase 2 │ Analysis Layer      → Trend Score + Creator Intelligence Score
Phase 3 │ NLP Layer           → sentence-transformers + KMeans Topic Clustering
Phase 4 │ Sentiment Analysis  → Hugging Face cardiffnlp/twitter-roberta-base-sentiment
Phase 5 │ Gemini Insight      → Google Gemini 1.5 Flash → Strengths / Opportunities / Ideas
Phase 6 │ Recommendation      → recommendation_score = trend×0.4 + sentiment×0.35 + engagement×0.25
```

---

## 📁 Project Structure

```
YTANALYSIS/
├── .env.example          ← API key template (copy → .env)
├── requirements.txt      ← All dependencies
├── config.py             ← Central config & score weights
│
├── db.py                 ← SQLite schema + upsert helpers
├── collector.py          ← YouTube API v3 data collection
│
├── analyzer.py           ← Trend Score, Creator Score, TF-IDF keywords
├── nlp.py                ← Phase 3: Topic extraction via embeddings
├── sentiment.py          ← Phase 4: HuggingFace sentiment pipeline
├── gemini_insight.py     ← Phase 5: Gemini AI insights
├── recommender.py        ← Phase 6: Content recommendation engine
│
├── main.py               ← CLI entry point
├── report.py             ← Rich terminal + HTML report generator
│
└── data/
    └── youtube.db        ← Auto-created SQLite database
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/MARSHALYADAV/YTANALYSIS.git
cd YTANALYSIS
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env`:
```env
YOUTUBE_API_KEY=AIzaSy...        # console.cloud.google.com → YouTube Data API v3
GEMINI_API_KEY=AIzaSy...         # aistudio.google.com/apikey (free)
GEMINI_MODEL=gemini-1.5-flash
```

### 3. Run

```bash
# Collect data (1 page = 50 videos, 5 pages = 250 videos)
python main.py collect --channel @mkbhd --pages 3

# Full report (terminal + HTML dashboard, auto-opens in browser)
python main.py report --channel @mkbhd
```

---

## 🛠️ All Commands

| Command | Description |
|---|---|
| `python main.py collect --channel @handle --pages N` | Fetch channel, videos & comments from YouTube API |
| `python main.py analyze --channel @handle` | Creator Intelligence Score breakdown |
| `python main.py trending --top 10` | Fastest-growing videos by Trend Score |
| `python main.py channels --top 5` | Channel leaderboard by Creator Score |
| `python main.py keywords --top 20` | TF-IDF trending keywords |
| `python main.py topics --channel @handle --source both` | Phase 3: NLP topic clusters |
| `python main.py sentiment --channel @handle` | Phase 4: Sentiment analysis |
| `python main.py report --channel @handle` | Full HTML report (all 6 phases) |
| `python main.py report --no-nlp` | Skip heavy NLP models (faster) |

---

## 📐 Scoring Formulas

### Phase 2 — Trend Score (per video)
```
trend_score =
    normalized(views)    × 0.40  +
    normalized(comments) × 0.20  +
    normalized(likes)    × 0.20  +
    recency_score        × 0.20

recency_score = 100 × 2^(−age_days / 30)   ← exponential decay, 30-day half-life
```

### Phase 2 — Creator Intelligence Score (per channel)
```
creator_score =
    engagement_rate      × 0.40   ← avg (likes + comments) / views
    upload_consistency   × 0.30   ← inverse std-dev of upload gaps
    growth_rate          × 0.30   ← recent vs older views-per-day ratio
```

| Range | Label |
|---|---|
| 85–100 | 🔥 Elite Creator |
| 70–84 | 🚀 High Potential |
| 50–69 | 📈 Growing |
| 0–49 | 🌱 Early Stage |

### Phase 6 — Recommendation Score (per topic)
```
recommendation_score =
    trend_score        × 0.40  +
    positive_sentiment × 0.35  +
    engagement_rate    × 0.25
```

---

## 🤖 Phase 5 — Gemini AI Insight Example

**Prompt built automatically from your DB:**
```
Channel: TechWithTim | Subscribers: 1,200,000 | Avg Views: 85,000
Top Topics: Python, AI Agents, Automation, Tutorial
Sentiment: Positive 74% | Neutral 19% | Negative 7%
```

**Gemini Response (structured):**
```
✅ Strengths
  • Strong engagement rate of 4.2% well above the 1-2% industry average...
  • Consistent upload schedule with < 3 days std-dev gap between videos...

⚠️ Weaknesses
  • Growth rate has plateaued — recent videos show 20% lower views-per-day...

🚀 Growth Opportunities
  • Shorts strategy: repurpose tutorial highlights for 3× reach expansion...

💡 Content Ideas
  1. "Build an AI Agent in 10 Minutes with Python" — matches top keyword cluster
  2. "n8n + Python Automation Full Course" — trending in engagement data
```

---

## 🗄️ Database Schema

```sql
channels      (channel_id, title, subscriber_count, view_count, video_count, …)
videos        (video_id, channel_id, title, published_at, view_count, like_count, comment_count, …)
top_comments  (comment_id, video_id, author, text, like_count, published_at)
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `google-api-python-client` | YouTube Data API v3 |
| `google-generativeai` | Gemini AI (Phase 5) |
| `sentence-transformers` | Text embeddings (Phase 3) |
| `scikit-learn` | KMeans clustering (Phase 3) |
| `transformers` + `torch` | Sentiment analysis (Phase 4) |
| `rich` | Beautiful terminal UI |
| `python-dotenv` | Environment variable management |

---

## 📊 YouTube API Quota (free: 10,000 units/day)

| Operation | Cost |
|---|---|
| Channel details | 1 unit |
| Video list (per page of 50) | 1 unit |
| Video details (per batch of 50) | 1 unit |
| Comment threads (per video) | 1 unit |
| Search (channel resolution) | 100 units |

**Tip:** Use `--channel UCxxxxxx` (direct channel ID) instead of `@handle` to save 100 quota units per run.

---

## 🎓 MCA Project — Built With

- **Python 3.10+** · SQLite · YouTube Data API v3
- **Hugging Face Transformers** · sentence-transformers · scikit-learn
- **Google Gemini 1.5 Flash** (free tier)
- **Rich** terminal UI · Chart.js HTML reports

---

*Built as an MCA project demonstrating end-to-end data engineering, NLP, and LLM integration.*
