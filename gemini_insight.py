"""
gemini_insight.py — Phase 5: Gemini AI Insight Layer
=====================================================
Uses the free-tier Gemini API (Google AI Studio) to generate
strategic LLM-powered insights for a YouTube channel.

The module builds a rich context prompt from real data in your DB,
sends it to Gemini, and returns a structured InsightReport.

Get your free API key at: https://aistudio.google.com/apikey
Model: gemini-1.5-flash (free tier, very generous limits)

Public API
----------
generate_channel_insights(channel_id)  → InsightReport dict
generate_topic_insights(topics, sentiment, channel_data)  → str (raw markdown)

InsightReport dict
------------------
{
  "channel_title": str,
  "prompt_context": str,      # the full context sent to Gemini
  "raw_response":  str,       # raw Gemini markdown output
  "sections": {
      "strengths":      list[str],
      "weaknesses":     list[str],
      "opportunities":  list[str],
      "content_ideas":  list[str],
  }
}
"""
from __future__ import annotations

import re
from typing import Optional

import config
import db
import analyzer

# ── Gemini client (lazy) ───────────────────────────────────────────────────────
_client = None

def _get_client():
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY or config.GEMINI_API_KEY == "YOUR_GEMINI_KEY_HERE":
            raise RuntimeError(
                "No Gemini API key found.\n"
                "  1. Get a free key at: https://aistudio.google.com/apikey\n"
                "  2. Set GEMINI_API_KEY=your_key in your .env file"
            )
        import google.generativeai as genai
        genai.configure(api_key=config.GEMINI_API_KEY)
        _client = genai.GenerativeModel(config.GEMINI_MODEL)
    return _client


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_prompt(
    channel_data: dict,
    top_videos: list[dict],
    keywords: list[tuple],
    sentiment: Optional[dict] = None,
    topics: Optional[list[dict]] = None,
) -> str:
    """Construct a rich, data-grounded prompt for Gemini."""

    # ── Channel stats block ─────────────────────────────────────────────────
    stats = (
        f"Channel Name     : {channel_data.get('channel_title', 'Unknown')}\n"
        f"Subscribers      : {channel_data.get('subscriber_count', 0):,}\n"
        f"Total Views      : {channel_data.get('total_views', 0):,}\n"
        f"Videos Analysed  : {channel_data.get('video_count', 0)}\n"
        f"Avg Views/Video  : {channel_data.get('avg_views', 0):,.0f}\n"
        f"Creator Score    : {channel_data.get('creator_score', 'N/A')}/100\n"
        f"Category         : {channel_data.get('label', 'N/A')}\n"
        f"Engagement Rate  : {channel_data.get('engagement_rate', 0):.1f}/100\n"
        f"Upload Consistency: {channel_data.get('upload_consistency', 0):.1f}/100\n"
        f"Growth Rate      : {channel_data.get('growth_rate', 0):.1f}/100"
    )

    # ── Top videos block ────────────────────────────────────────────────────
    vid_lines = []
    for i, v in enumerate(top_videos[:5], 1):
        vid_lines.append(
            f"  {i}. \"{v['title'][:60]}\" "
            f"(Views: {v['view_count']:,}, "
            f"Likes: {v['like_count']:,}, "
            f"Trend Score: {v.get('trend_score', 0):.1f})"
        )
    videos_block = "\n".join(vid_lines) if vid_lines else "  No video data."

    # ── Top keywords ────────────────────────────────────────────────────────
    kw_str = ", ".join(w for w, _ in keywords[:10]) if keywords else "N/A"

    # ── Sentiment block ─────────────────────────────────────────────────────
    sent_block = ""
    if sentiment and sentiment.get("total", 0) > 0:
        sent_block = (
            f"\nAudience Sentiment ({sentiment['total']} comments analysed):\n"
            f"  Positive : {sentiment['positive']['pct']}%\n"
            f"  Neutral  : {sentiment['neutral']['pct']}%\n"
            f"  Negative : {sentiment['negative']['pct']}%"
        )

    # ── Topic clusters ──────────────────────────────────────────────────────
    topic_block = ""
    if topics:
        topic_lines = [f"  • {t['label']} ({t['size']} videos/comments)" for t in topics[:6]]
        topic_block = "\nIdentified Content Topics:\n" + "\n".join(topic_lines)

    prompt = f"""You are a YouTube channel strategy consultant with deep expertise in content analytics, audience growth, and creator monetization.

Analyse the following YouTube channel data and provide a comprehensive strategic report.

═══════════════════════════════════════
CHANNEL DATA
═══════════════════════════════════════
{stats}

Top Performing Videos:
{videos_block}

Top Keywords in Content: {kw_str}
{sent_block}
{topic_block}

═══════════════════════════════════════
REQUIRED OUTPUT FORMAT
═══════════════════════════════════════

Respond ONLY in this exact structure. Be specific, data-driven, and actionable.
Use the actual numbers and topics from the data above.

## ✅ Strengths
- [Specific strength based on the data, 1-2 sentences each]
- [Strength 2]
- [Strength 3]

## ⚠️ Weaknesses
- [Specific weakness with supporting data]
- [Weakness 2]
- [Weakness 3]

## 🚀 Growth Opportunities
- [Specific, actionable opportunity with estimated impact]
- [Opportunity 2]
- [Opportunity 3]
- [Opportunity 4]

## 💡 Recommended Content Ideas
1. [Specific video title idea] — [Why this would perform well based on the data]
2. [Content idea 2] — [Rationale]
3. [Content idea 3] — [Rationale]
4. [Content idea 4] — [Rationale]
5. [Content idea 5] — [Rationale]

## 📊 Executive Summary
[2-3 sentence high-level summary of the channel's position and top priority action]
"""
    return prompt


# ── Response parser ────────────────────────────────────────────────────────────

def _parse_sections(text: str) -> dict:
    """
    Extract structured lists from the Gemini markdown response.
    Returns dict with keys: strengths, weaknesses, opportunities, content_ideas, summary.
    """
    def _extract_section(header_pattern: str) -> list[str]:
        pattern = rf"{header_pattern}\s*\n(.*?)(?=\n##|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if not match:
            return []
        block = match.group(1).strip()
        items = []
        for line in block.splitlines():
            line = line.strip()
            # Remove leading bullet/number markers
            line = re.sub(r'^[-*•]\s*', '', line)
            line = re.sub(r'^\d+\.\s*', '', line)
            if line:
                items.append(line)
        return items

    return {
        "strengths":     _extract_section(r"##\s*✅?\s*Strengths?"),
        "weaknesses":    _extract_section(r"##\s*⚠️?\s*Weaknesses?"),
        "opportunities": _extract_section(r"##\s*🚀?\s*Growth Opportunities?"),
        "content_ideas": _extract_section(r"##\s*💡?\s*Recommended Content Ideas?"),
        "summary":       _extract_section(r"##\s*📊?\s*Executive Summary"),
    }


# ── Main public function ───────────────────────────────────────────────────────

def generate_channel_insights(
    channel_id: str,
    sentiment_data: Optional[dict] = None,
    topic_data: Optional[list[dict]] = None,
) -> dict:
    """
    Build context from DB, call Gemini, parse and return InsightReport.

    Parameters
    ----------
    channel_id     : YouTube channel ID (must already be in DB)
    sentiment_data : Result from sentiment.analyze_comments() — optional
    topic_data     : Result from nlp.extract_topics_from_titles() — optional
    """
    from rich.console import Console
    from rich.spinner import Spinner
    console = Console()

    # ── Gather data from DB ───────────────────────────────────────────────
    channel_row = db.get_channel(channel_id)
    if not channel_row:
        return {"error": f"Channel {channel_id} not in database. Run 'collect' first."}

    creator_score = analyzer.compute_creator_score(channel_id)
    videos_scored = analyzer.compute_trend_scores(db.get_videos_for_channel(channel_id))
    keywords      = analyzer.trending_keywords(10)

    # Average views
    all_videos  = db.get_videos_for_channel(channel_id)
    avg_views   = (
        sum(int(v["view_count"]) for v in all_videos) / len(all_videos)
        if all_videos else 0
    )

    channel_data = {
        "channel_title":     channel_row["title"],
        "subscriber_count":  channel_row["subscriber_count"],
        "total_views":       channel_row["view_count"],
        "video_count":       len(all_videos),
        "avg_views":         avg_views,
        "creator_score":     creator_score.get("creator_score", "N/A"),
        "label":             creator_score.get("label", "N/A"),
        "engagement_rate":   creator_score.get("engagement_rate", 0),
        "upload_consistency":creator_score.get("upload_consistency", 0),
        "growth_rate":       creator_score.get("growth_rate", 0),
    }

    # ── Build prompt ──────────────────────────────────────────────────────
    prompt = _build_prompt(
        channel_data=channel_data,
        top_videos=videos_scored[:5],
        keywords=keywords,
        sentiment=sentiment_data,
        topics=topic_data,
    )

    # ── Call Gemini ───────────────────────────────────────────────────────
    console.print(
        f"[bold cyan]🤖 Calling Gemini ({config.GEMINI_MODEL}) for channel insights…[/bold cyan]"
    )
    try:
        client   = _get_client()
        response = client.generate_content(prompt)
        raw_text = response.text
    except Exception as e:
        return {"error": f"Gemini API error: {e}"}

    sections = _parse_sections(raw_text)

    return {
        "channel_title":  channel_data["channel_title"],
        "channel_data":   channel_data,
        "prompt_context": prompt,
        "raw_response":   raw_text,
        "sections":       sections,
    }
