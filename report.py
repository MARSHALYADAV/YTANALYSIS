"""
report.py — Rich terminal output + HTML report generation.
Covers Phases 1-6: analytics, NLP topics, sentiment, Gemini insights, recommendations.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.columns import Columns

import config
import analyzer
import db

console = Console()


# ── Terminal helpers ───────────────────────────────────────────────────────────

def _fmt_number(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def _score_bar(score: float, width: int = 20) -> str:
    filled = int(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _score_color(score: float) -> str:
    if score >= 85:
        return "bold red"
    if score >= 70:
        return "bold yellow"
    if score >= 50:
        return "bold green"
    return "dim"


# ── Creator Score Panel ────────────────────────────────────────────────────────

def print_creator_report(channel_id: str) -> None:
    result = analyzer.compute_creator_score(channel_id)
    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        return

    score = result["creator_score"]
    label = result["label"]
    color = _score_color(score)

    console.print()
    console.rule(f"[bold cyan]🎬 Creator Intelligence Report")
    console.print()

    # Score panel
    bar = _score_bar(score)
    score_text = Text()
    score_text.append(f"\n  Channel  : {result['channel_title']}\n", style="bold white")
    score_text.append(f"  Subs     : {_fmt_number(result['subscriber_count'])}\n", style="white")
    score_text.append(f"  Videos   : {result['video_count']}\n\n", style="white")
    score_text.append(f"  Score    : ", style="bold white")
    score_text.append(f"{score}/100\n", style=color)
    score_text.append(f"  Category : {label}\n\n", style=color)
    score_text.append(f"  [{bar}]\n\n", style=color)
    score_text.append(f"  Engagement    : {result['engagement_rate']:>5.1f}/100\n", style="cyan")
    score_text.append(f"  Consistency   : {result['upload_consistency']:>5.1f}/100\n", style="cyan")
    score_text.append(f"  Growth Rate   : {result['growth_rate']:>5.1f}/100\n", style="cyan")

    console.print(Panel(score_text, title="[bold]Creator Score", border_style="cyan"))
    console.print()


# ── Trending Videos Table ──────────────────────────────────────────────────────

def print_trending_videos(top_n: int = 10) -> None:
    videos = analyzer.fastest_growing_videos(top_n)
    if not videos:
        console.print("[yellow]No video data found. Run 'collect' first.[/yellow]")
        return

    console.print()
    console.rule("[bold cyan]🔥 Fastest Growing Videos — Trend Score Ranking")

    table = Table(box=box.ROUNDED, show_lines=True, style="on grey7")
    table.add_column("#",          style="dim",       width=3,  justify="right")
    table.add_column("Title",      style="bold white", width=38)
    table.add_column("Trend",      style="bold red",  width=7,  justify="right")
    table.add_column("Views",      style="cyan",      width=8,  justify="right")
    table.add_column("Likes",      style="green",     width=8,  justify="right")
    table.add_column("Comments",   style="yellow",    width=9,  justify="right")
    table.add_column("Recency",    style="magenta",   width=9,  justify="right")
    table.add_column("Published",  style="dim",       width=12)

    for i, v in enumerate(videos, 1):
        pub = (v["published_at"] or "")[:10]
        title = v["title"][:35] + ("…" if len(v["title"]) > 35 else "")
        table.add_row(
            str(i),
            title,
            f"{v['trend_score']:.1f}",
            _fmt_number(v["view_count"]),
            _fmt_number(v["like_count"]),
            _fmt_number(v["comment_count"]),
            f"{v['recency_score']:.1f}",
            pub,
        )

    console.print(table)
    console.print()


# ── Channel Leaderboard ────────────────────────────────────────────────────────

def print_channel_leaderboard(top_n: int = 10) -> None:
    channels = analyzer.fastest_growing_channels(top_n)
    if not channels:
        console.print("[yellow]No channel data. Run 'collect' first.[/yellow]")
        return

    console.print()
    console.rule("[bold cyan]🚀 Fastest Growing Channels")

    table = Table(box=box.ROUNDED, show_lines=True, style="on grey7")
    table.add_column("#",           style="dim",       width=3,  justify="right")
    table.add_column("Channel",     style="bold white", width=28)
    table.add_column("Score",       style="bold red",  width=7,  justify="right")
    table.add_column("Category",    style="bold",      width=18)
    table.add_column("Subs",        style="cyan",      width=8,  justify="right")
    table.add_column("Engagement",  style="green",     width=11, justify="right")
    table.add_column("Consistency", style="yellow",    width=12, justify="right")
    table.add_column("Growth",      style="magenta",   width=8,  justify="right")

    for i, ch in enumerate(channels, 1):
        table.add_row(
            str(i),
            ch["channel_title"][:25],
            f"{ch['creator_score']:.1f}",
            ch["label"],
            _fmt_number(ch["subscriber_count"]),
            f"{ch['engagement_rate']:.1f}",
            f"{ch['upload_consistency']:.1f}",
            f"{ch['growth_rate']:.1f}",
        )

    console.print(table)
    console.print()


# ── Trending Keywords ──────────────────────────────────────────────────────────

def print_trending_keywords(top_n: int = 20) -> None:
    kws = analyzer.trending_keywords(top_n)
    if not kws:
        console.print("[yellow]No video data. Run 'collect' first.[/yellow]")
        return

    console.print()
    console.rule("[bold cyan]📊 Trending Keywords (TF-IDF)")

    max_score = kws[0][1] if kws else 1
    table = Table(box=box.SIMPLE_HEAVY, style="on grey7")
    table.add_column("Rank", style="dim",       width=5,  justify="right")
    table.add_column("Keyword", style="bold white", width=20)
    table.add_column("Score",  style="cyan",    width=8,  justify="right")
    table.add_column("Bar",    style="green",   width=30)

    for i, (word, score) in enumerate(kws, 1):
        bar_width = int(score / max_score * 28)
        bar = "█" * bar_width
        table.add_row(str(i), word, f"{score:.1f}", bar)

    console.print(table)
    console.print()


# ── Phase 3: NLP Topics ──────────────────────────────────────────────────────

def print_topics(
    channel_id: Optional[str] = None,
    source: str = "both",
    n_topics: int = 6,
) -> None:
    from nlp import extract_topics_from_titles, extract_topics_from_comments

    console.print()
    console.rule("[bold cyan]🧠 Phase 3 — NLP Topic Clusters")

    all_topics: list[dict] = []

    if source in ("titles", "both"):
        console.print("[dim]Embedding video titles…[/dim]")
        title_topics = extract_topics_from_titles(channel_id, n_topics=n_topics)
        if title_topics:
            _print_topic_table(title_topics, "Video Title Topics")
            all_topics.extend(title_topics)

    if source in ("comments", "both"):
        console.print("[dim]Embedding comment text…[/dim]")
        comment_topics = extract_topics_from_comments(channel_id, n_topics=n_topics)
        if comment_topics:
            _print_topic_table(comment_topics, "Comment Topics")
            all_topics.extend(comment_topics)

    if not all_topics:
        console.print("[yellow]No text data for topic extraction. Run 'collect' first.[/yellow]")


def _print_topic_table(topics: list[dict], title: str) -> None:
    console.print(f"\n  [bold white]{title}[/bold white]")
    table = Table(box=box.ROUNDED, show_lines=True, style="on grey7")
    table.add_column("#",        style="dim",        width=3,  justify="right")
    table.add_column("Topic",    style="bold yellow", width=22)
    table.add_column("Keywords", style="cyan",        width=40)
    table.add_column("Size",     style="green",       width=6,  justify="right")
    table.add_column("Example",  style="dim",         width=45)

    for i, t in enumerate(topics, 1):
        kws  = ", ".join(t["keywords"][:4])
        ex   = t["examples"][0][:42] + "…" if t["examples"] else ""
        table.add_row(str(i), t["label"], kws, str(t["size"]), ex)

    console.print(table)
    console.print()


# ── Phase 4: Sentiment ────────────────────────────────────────────────────────

def print_sentiment(
    channel_id: Optional[str] = None,
    source: str = "comments",
) -> dict:
    """Run sentiment analysis and print results. Returns the result dict."""
    from sentiment import analyze_comments, analyze_titles

    console.print()
    console.rule("[bold cyan]📊 Phase 4 — Sentiment Analysis")
    console.print(f"[dim]Analysing {source}…[/dim]")

    result = analyze_comments(channel_id) if source == "comments" else analyze_titles(channel_id)

    if result["total"] == 0:
        console.print("[yellow]No text data found. Run 'collect' with comments enabled.[/yellow]")
        return result

    pos = result["positive"]["pct"]
    neu = result["neutral"]["pct"]
    neg = result["negative"]["pct"]
    total = result["total"]

    # Build colour-coded panel
    text = Text()
    text.append(f"\n  Analysed : {total} {source}\n\n", style="dim")
    text.append(f"  Positive : ", style="bold white")
    text.append(f"{pos}%", style="bold green")
    text.append(f"  {_score_bar(pos, 30)}\n", style="green")
    text.append(f"  Neutral  : ", style="bold white")
    text.append(f"{neu}%", style="bold yellow")
    text.append(f"  {_score_bar(neu, 30)}\n", style="yellow")
    text.append(f"  Negative : ", style="bold white")
    text.append(f"{neg}%", style="bold red")
    text.append(f"  {_score_bar(neg, 30)}\n", style="red")

    # Sample comments
    for label, color in (("positive", "green"), ("negative", "red")):
        samples = result["samples"].get(label, [])
        if samples:
            text.append(f"\n  Sample {label.capitalize()} Comment:\n", style=f"bold {color}")
            text.append(f"    “{samples[0][:100]}…”\n", style="dim")

    console.print(Panel(text, title="[bold]📊 Audience Sentiment", border_style="blue"))
    console.print()
    return result


# ── Phase 5: Gemini Insights ──────────────────────────────────────────────────

def print_gemini_insights(
    channel_id: str,
    sentiment_data: Optional[dict] = None,
    topic_data: Optional[list] = None,
) -> dict:
    """Call Gemini and pretty-print the structured insight report."""
    from gemini_insight import generate_channel_insights

    console.print()
    console.rule("[bold cyan]🤖 Phase 5 — Gemini AI Insight Engine")

    insight = generate_channel_insights(
        channel_id,
        sentiment_data=sentiment_data,
        topic_data=topic_data,
    )

    if "error" in insight:
        console.print(f"[red]{insight['error']}[/red]")
        return insight

    sections = insight["sections"]

    def _section_panel(icon: str, title: str, items: list[str], color: str):
        if not items:
            return
        text = Text("\n")
        for item in items:
            text.append(f"  • ", style=f"bold {color}")
            text.append(f"{item}\n", style="white")
        console.print(Panel(text, title=f"[bold {color}]{icon} {title}", border_style=color))
        console.print()

    _section_panel("✅", "Strengths",            sections["strengths"],     "green")
    _section_panel("⚠️", "Weaknesses",           sections["weaknesses"],    "yellow")
    _section_panel("🚀", "Growth Opportunities", sections["opportunities"], "cyan")
    _section_panel("💡", "Content Ideas",        sections["content_ideas"], "magenta")

    summary = sections.get("summary", [])
    if summary:
        console.print(Panel(
            Text("\n  " + "\n  ".join(summary) + "\n"),
            title="[bold white]📊 Executive Summary",
            border_style="white",
        ))
    console.print()
    return insight


# ── Phase 6: Recommendations ───────────────────────────────────────────────────

def print_recommendations(
    channel_id: str,
    topics: Optional[list] = None,
    sentiment_result: Optional[dict] = None,
    top_n: int = 10,
) -> list[dict]:
    """Print Phase 6 ranked content recommendations."""
    from recommender import recommend_topics, quick_recommend_from_keywords

    console.print()
    console.rule("[bold cyan]🎯 Phase 6 — Content Recommendation Engine")
    console.print(f"[dim]score = trend×0.40 + sentiment×0.35 + engagement×0.25[/dim]\n")

    if topics:
        recs = recommend_topics(
            channel_id, topics, sentiment_result, top_n=top_n
        )
    else:
        kws = analyzer.trending_keywords(top_n)
        recs = quick_recommend_from_keywords(channel_id, kws, sentiment_result, top_n=top_n)

    if not recs:
        console.print("[yellow]No recommendation data. Run 'collect' first.[/yellow]")
        return []

    table = Table(box=box.ROUNDED, show_lines=True, style="on grey7")
    table.add_column("#",          style="dim",        width=3,  justify="right")
    table.add_column("Topic",      style="bold yellow", width=22)
    table.add_column("Score",      style="bold red",   width=7,  justify="right")
    table.add_column("Trend",      style="cyan",        width=7,  justify="right")
    table.add_column("Sentiment",  style="green",       width=10, justify="right")
    table.add_column("Engagement", style="magenta",     width=11, justify="right")
    table.add_column("Keywords",   style="dim",         width=30)

    for r in recs:
        table.add_row(
            str(r["rank"]),
            r["topic"],
            f"{r['score']:.1f}",
            f"{r['trend_component']:.1f}",
            f"{r['sent_component']:.1f}",
            f"{r['eng_component']:.1f}",
            ", ".join(r["keywords"][:4]),
        )

    console.print(table)
    console.print()
    return recs


# ── HTML Report ────────────────────────────────────────────────────────────────

def generate_html_report(channel_id: Optional[str] = None) -> Path:
    """
    Generate a self-contained HTML report with all 6 phases.
    Returns the path to the saved file.
    """
    # ── Gather data ───────────────────────────────────────────────────────────
    if channel_id:
        creator = analyzer.compute_creator_score(channel_id)
        videos  = analyzer.compute_trend_scores(db.get_videos_for_channel(channel_id))[:15]
        title   = creator.get("channel_title", channel_id)
    else:
        creator = None
        videos  = analyzer.fastest_growing_videos(15)
        title   = "All Channels"

    keywords = analyzer.trending_keywords(15)
    channels = analyzer.fastest_growing_channels(10)

    # Build JSON data blobs for Chart.js
    kw_labels  = json.dumps([k for k, _ in keywords])
    kw_scores  = json.dumps([round(s, 1) for _, s in keywords])

    vid_labels = json.dumps([v["title"][:40] for v in videos])
    vid_scores = json.dumps([v["trend_score"] for v in videos])
    vid_views  = json.dumps([v["view_count"] for v in videos])

    ch_labels  = json.dumps([ch["channel_title"][:25] for ch in channels])
    ch_scores  = json.dumps([ch["creator_score"] for ch in channels])

    creator_html = ""
    if creator and "error" not in creator:
        s = creator["creator_score"]
        color = "#ef4444" if s >= 85 else "#eab308" if s >= 70 else "#22c55e" if s >= 50 else "#6b7280"
        creator_html = f"""
        <div class="creator-card">
          <div class="creator-header">
            <h2>🎬 {creator['channel_title']}</h2>
            <span class="sub-count">{_fmt_number(creator['subscriber_count'])} subscribers</span>
          </div>
          <div class="score-ring-wrapper">
            <div class="score-ring" style="--score:{s};--color:{color};">
              <span class="score-num" style="color:{color}">{s}</span>
              <span class="score-denom">/100</span>
            </div>
            <div class="score-label" style="color:{color}">{creator['label']}</div>
          </div>
          <div class="component-grid">
            <div class="comp">
              <div class="comp-label">Engagement</div>
              <div class="comp-bar"><div style="width:{creator['engagement_rate']}%;background:#22c55e"></div></div>
              <div class="comp-val">{creator['engagement_rate']:.1f}</div>
            </div>
            <div class="comp">
              <div class="comp-label">Consistency</div>
              <div class="comp-bar"><div style="width:{creator['upload_consistency']}%;background:#3b82f6"></div></div>
              <div class="comp-val">{creator['upload_consistency']:.1f}</div>
            </div>
            <div class="comp">
              <div class="comp-label">Growth Rate</div>
              <div class="comp-bar"><div style="width:{creator['growth_rate']}%;background:#a855f7"></div></div>
              <div class="comp-val">{creator['growth_rate']:.1f}</div>
            </div>
          </div>
        </div>
        """

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YouTube Analytics — {title}</title>
<meta name="description" content="YouTube channel analytics report with trend scores and creator intelligence metrics.">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:        #0a0a0f;
    --surface:   #12121a;
    --surface2:  #1a1a26;
    --border:    #2a2a40;
    --text:      #e8e8f0;
    --text-dim:  #8888aa;
    --accent:    #6366f1;
    --accent2:   #8b5cf6;
    --red:       #ef4444;
    --yellow:    #eab308;
    --green:     #22c55e;
    --cyan:      #06b6d4;
  }}

  body {{
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    line-height: 1.6;
  }}

  /* Header */
  header {{
    background: linear-gradient(135deg, #0d0d1a 0%, #1a0a2e 50%, #0a1a2e 100%);
    border-bottom: 1px solid var(--border);
    padding: 2rem 3rem;
    position: relative;
    overflow: hidden;
  }}
  header::before {{
    content: '';
    position: absolute;
    top: -50%;
    left: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(99,102,241,0.15) 0%, transparent 70%);
    pointer-events: none;
  }}
  .header-inner {{
    max-width: 1400px;
    margin: 0 auto;
    position: relative;
  }}
  header h1 {{
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #6366f1, #a855f7, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  header .subtitle {{
    color: var(--text-dim);
    font-size: 0.9rem;
    margin-top: 0.25rem;
  }}
  .generated {{
    position: absolute;
    top: 0;
    right: 0;
    font-size: 0.75rem;
    color: var(--text-dim);
  }}

  /* Main layout */
  main {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem 3rem;
    display: grid;
    gap: 2rem;
  }}

  /* Cards */
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
  }}
  .card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.5), transparent);
  }}
  .card h2 {{
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1.25rem;
  }}
  .card h2 span {{
    color: var(--text);
  }}

  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 2rem; }}

  /* Creator card */
  .creator-card {{
    background: linear-gradient(135deg, var(--surface) 0%, #1a1030 100%);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 16px;
    padding: 2rem;
    position: relative;
    overflow: hidden;
  }}
  .creator-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6366f1, #a855f7, #06b6d4);
  }}
  .creator-header {{ margin-bottom: 1.5rem; }}
  .creator-header h2 {{ font-size: 1.5rem; font-weight: 700; color: var(--text); }}
  .sub-count {{ color: var(--text-dim); font-size: 0.875rem; }}
  .score-ring-wrapper {{
    display: flex;
    align-items: center;
    gap: 2rem;
    margin-bottom: 2rem;
  }}
  .score-ring {{
    width: 120px;
    height: 120px;
    border-radius: 50%;
    background: conic-gradient(var(--color) calc(var(--score) * 3.6deg), #2a2a40 0deg);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    position: relative;
    flex-shrink: 0;
  }}
  .score-ring::before {{
    content: '';
    position: absolute;
    width: 90px;
    height: 90px;
    border-radius: 50%;
    background: var(--surface);
  }}
  .score-num {{
    font-size: 1.75rem;
    font-weight: 800;
    position: relative;
    z-index: 1;
  }}
  .score-denom {{
    font-size: 0.75rem;
    color: var(--text-dim);
    position: relative;
    z-index: 1;
  }}
  .score-label {{
    font-size: 1.1rem;
    font-weight: 700;
  }}
  .component-grid {{ display: grid; gap: 0.75rem; }}
  .comp {{ display: grid; grid-template-columns: 110px 1fr 50px; gap: 0.75rem; align-items: center; }}
  .comp-label {{ font-size: 0.85rem; color: var(--text-dim); }}
  .comp-bar {{
    height: 8px;
    background: var(--surface2);
    border-radius: 4px;
    overflow: hidden;
  }}
  .comp-bar div {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease;
  }}
  .comp-val {{ font-size: 0.85rem; font-weight: 600; text-align: right; }}

  /* Table */
  .data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
  }}
  .data-table th {{
    background: var(--surface2);
    color: var(--text-dim);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.75rem;
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  .data-table td {{
    padding: 0.75rem 1rem;
    border-bottom: 1px solid rgba(42,42,64,0.5);
    vertical-align: middle;
  }}
  .data-table tr:last-child td {{ border-bottom: none; }}
  .data-table tr:hover td {{ background: rgba(99,102,241,0.05); }}
  .rank {{ color: var(--text-dim); font-weight: 700; }}
  .score-cell {{ font-weight: 700; }}
  .score-high  {{ color: var(--red); }}
  .score-mid   {{ color: var(--yellow); }}
  .score-low   {{ color: var(--green); }}
  .views-cell  {{ color: var(--cyan); }}
  .num-dim     {{ color: var(--text-dim); }}

  /* Sentiment bars */
  .sentiment-grid {{ display: grid; gap: 0.75rem; }}
  .sent-bar-wrap {{ display: grid; grid-template-columns: 80px 1fr 55px; gap: 0.75rem; align-items: center; }}
  .sent-label {{ font-size: 0.85rem; color: var(--text-dim); }}
  .sent-bar {{ height: 10px; background: var(--surface2); border-radius: 5px; overflow: hidden; }}
  .sent-bar div {{ height: 100%; border-radius: 5px; transition: width 1s ease; }}
  .sent-pct {{ font-size: 0.9rem; font-weight: 700; text-align: right; }}

  /* Gemini card */
  .gemini-card {{ border-color: rgba(99,102,241,0.4); }}
  .gemini-card::before {{ background: linear-gradient(90deg, #6366f1, #a855f7); }}
  .gemini-summary {{
    color: var(--text-dim);
    font-size: 0.925rem;
    line-height: 1.7;
    margin-bottom: 1.5rem;
    padding: 1rem;
    background: var(--surface2);
    border-radius: 8px;
    border-left: 3px solid var(--accent);
  }}
  .gemini-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }}
  .gemini-section {{ background: var(--surface2); border-radius: 10px; padding: 1rem; }}
  .gsec-title {{ font-size: 0.85rem; font-weight: 700; margin-bottom: 0.75rem; color: var(--text); }}
  .gemini-section ul {{ list-style: none; padding: 0; margin: 0; }}
  .gemini-section li {{ font-size: 0.85rem; color: var(--text-dim); padding: 0.35rem 0; border-bottom: 1px solid rgba(255,255,255,0.04); line-height: 1.5; }}
  .gemini-section li:last-child {{ border-bottom: none; }}
  .gemini-strengths   {{ border-top: 2px solid #22c55e; }}
  .gemini-weaknesses  {{ border-top: 2px solid #eab308; }}
  .gemini-opps        {{ border-top: 2px solid #06b6d4; }}
  .gemini-ideas       {{ border-top: 2px solid #a855f7; }}

  canvas {{ max-height: 320px; }}

  /* Footer */
  footer {{
    text-align: center;
    padding: 2rem;
    color: var(--text-dim);
    font-size: 0.8rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
  }}

  @media (max-width: 900px) {{
    main {{ padding: 1rem; }}
    .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
    header {{ padding: 1.5rem; }}
  }}
</style>
</head>
<body>

<header>
  <div class="header-inner">
    <h1>📊 YouTube Analytics</h1>
    <div class="subtitle">Channel: <strong>{title}</strong> &nbsp;·&nbsp; Intelligence Report</div>
    <div class="generated">Generated {now}</div>
  </div>
</header>

<main>

  {creator_html}

  {sentiment_html}
  {gemini_html}
  {reco_html}

  <!-- Trend Score Chart -->
  <div class="card">
    <h2>🔥 <span>Top Videos by Trend Score</span></h2>
    <canvas id="trendChart"></canvas>
  </div>

  <div class="grid-2">
    <!-- Keywords Chart -->
    <div class="card">
      <h2>📊 <span>Trending Keywords</span></h2>
      <canvas id="kwChart"></canvas>
    </div>
    <!-- Channel Leaderboard Chart -->
    <div class="card">
      <h2>🚀 <span>Channel Creator Scores</span></h2>
      <canvas id="chChart"></canvas>
    </div>
  </div>

  <!-- Top Videos Table -->
  <div class="card">
    <h2>📋 <span>Video Details</span></h2>
    <table class="data-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Title</th>
          <th>Trend Score</th>
          <th>Views</th>
          <th>Likes</th>
          <th>Comments</th>
          <th>Recency</th>
          <th>Published</th>
        </tr>
      </thead>
      <tbody>
        {"".join(f'''
        <tr>
          <td class="rank">{i}</td>
          <td>{v["title"][:60]}</td>
          <td class="score-cell {'score-high' if v['trend_score']>=70 else 'score-mid' if v['trend_score']>=40 else 'score-low'}">{v['trend_score']}</td>
          <td class="views-cell">{_fmt_number(v['view_count'])}</td>
          <td class="num-dim">{_fmt_number(v['like_count'])}</td>
          <td class="num-dim">{_fmt_number(v['comment_count'])}</td>
          <td class="num-dim">{v['recency_score']:.1f}</td>
          <td class="num-dim">{(v['published_at'] or '')[:10]}</td>
        </tr>''' for i, v in enumerate(videos, 1))}
      </tbody>
    </table>
  </div>

  <!-- Channel Leaderboard Table -->
  <div class="card">
    <h2>🏆 <span>Channel Leaderboard</span></h2>
    <table class="data-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Channel</th>
          <th>Creator Score</th>
          <th>Category</th>
          <th>Subscribers</th>
          <th>Engagement</th>
          <th>Consistency</th>
          <th>Growth</th>
        </tr>
      </thead>
      <tbody>
        {"".join(f'''
        <tr>
          <td class="rank">{i}</td>
          <td>{ch["channel_title"]}</td>
          <td class="score-cell {'score-high' if ch['creator_score']>=70 else 'score-mid' if ch['creator_score']>=50 else 'score-low'}">{ch['creator_score']}</td>
          <td>{ch['label']}</td>
          <td class="views-cell">{_fmt_number(ch['subscriber_count'])}</td>
          <td class="num-dim">{ch['engagement_rate']:.1f}</td>
          <td class="num-dim">{ch['upload_consistency']:.1f}</td>
          <td class="num-dim">{ch['growth_rate']:.1f}</td>
        </tr>''' for i, ch in enumerate(channels, 1))}
      </tbody>
    </table>
  </div>

</main>

<footer>YouTube Analytics System &nbsp;·&nbsp; Built with Python + YouTube Data API v3</footer>

<script>
const chartDefaults = {{
  color: '#8888aa',
  borderColor: '#2a2a40',
  font: {{ family: 'Inter', size: 12 }},
}};
Chart.defaults.color = chartDefaults.color;
Chart.defaults.borderColor = chartDefaults.borderColor;
Chart.defaults.font = chartDefaults.font;

// Trend Score Bar Chart
new Chart(document.getElementById('trendChart'), {{
  type: 'bar',
  data: {{
    labels: {vid_labels},
    datasets: [
      {{
        label: 'Trend Score',
        data: {vid_scores},
        backgroundColor: 'rgba(99,102,241,0.85)',
        borderRadius: 6,
        yAxisID: 'y',
      }},
      {{
        label: 'Views (×1000)',
        data: {vid_views}.map(v => v/1000),
        backgroundColor: 'rgba(6,182,212,0.4)',
        borderRadius: 6,
        yAxisID: 'y2',
        type: 'bar',
      }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ position: 'top' }} }},
    scales: {{
      x: {{ grid: {{ color: '#1a1a26' }}, ticks: {{ maxRotation: 45 }} }},
      y: {{ grid: {{ color: '#1a1a26' }}, max: 100, title: {{ display: true, text: 'Trend Score' }} }},
      y2: {{ position: 'right', grid: {{ drawOnChartArea: false }}, title: {{ display: true, text: 'Views (K)' }} }},
    }}
  }}
}});

// Keywords Horizontal Bar
new Chart(document.getElementById('kwChart'), {{
  type: 'bar',
  data: {{
    labels: {kw_labels},
    datasets: [{{
      label: 'TF-IDF Score',
      data: {kw_scores},
      backgroundColor: 'rgba(168,85,247,0.8)',
      borderRadius: 4,
    }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ grid: {{ color: '#1a1a26' }} }},
      y: {{ grid: {{ color: '#1a1a26' }} }},
    }}
  }}
}});

// Channel Scores
new Chart(document.getElementById('chChart'), {{
  type: 'radar',
  data: {{
    labels: {ch_labels},
    datasets: [{{
      label: 'Creator Score',
      data: {ch_scores},
      backgroundColor: 'rgba(34,197,94,0.2)',
      borderColor: '#22c55e',
      pointBackgroundColor: '#22c55e',
      pointRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    scales: {{
      r: {{
        min: 0, max: 100,
        grid: {{ color: '#2a2a40' }},
        ticks: {{ display: false }},
        angleLines: {{ color: '#2a2a40' }},
        pointLabels: {{ color: '#8888aa', font: {{ size: 11 }} }},
      }}
    }},
    plugins: {{ legend: {{ display: false }} }},
  }}
}});

// Sentiment Doughnut (Phase 4)
if (document.getElementById('sentChart')) {{
  new Chart(document.getElementById('sentChart'), {{
    type: 'doughnut',
    data: {{
      labels: ['Positive', 'Neutral', 'Negative'],
      datasets: [{{
        data: [{sentiment_result['positive']['pct'] if sentiment_result and sentiment_result.get('total') else 0},
               {sentiment_result['neutral']['pct']  if sentiment_result and sentiment_result.get('total') else 0},
               {sentiment_result['negative']['pct'] if sentiment_result and sentiment_result.get('total') else 0}],
        backgroundColor: ['rgba(34,197,94,0.8)', 'rgba(234,179,8,0.8)', 'rgba(239,68,68,0.8)'],
        borderWidth: 0,
        hoverOffset: 8,
      }}]
    }},
    options: {{
      responsive: true,
      cutout: '65%',
      plugins: {{ legend: {{ position: 'right' }} }},
    }}
  }});
}}

// Recommendation Score Chart (Phase 6)
if (document.getElementById('recoChart') && {reco_labels}.length > 0) {{
  new Chart(document.getElementById('recoChart'), {{
    type: 'bar',
    data: {{
      labels: {reco_labels},
      datasets: [{{
        label: 'Recommendation Score',
        data: {reco_scores},
        backgroundColor: [
          'rgba(99,102,241,0.9)', 'rgba(139,92,246,0.85)', 'rgba(168,85,247,0.80)',
          'rgba(192,78,248,0.75)', 'rgba(217,70,239,0.70)', 'rgba(236,72,153,0.65)',
          'rgba(244,63,94,0.60)',  'rgba(249,115,22,0.55)', 'rgba(234,179,8,0.50)',
          'rgba(34,197,94,0.45)',
        ],
        borderRadius: 6,
      }}]
    }},
    options: {{
      indexAxis: 'y',
      responsive: true,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ grid: {{ color: '#1a1a26' }}, min: 0, max: 100 }},
        y: {{ grid: {{ color: '#1a1a26' }} }},
      }}
    }}
  }});
}}
</script>
</body>
</html>"""

    out_path = config.REPORTS_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path
