"""
main.py — CLI entry point for YouTube Analytics System.

Commands
--------
  collect    Fetch channel + video + comment data from YouTube API
  analyze    Print creator intelligence report for a channel
  trending   Show fastest-growing videos by trend score
  channels   Show fastest-growing channels by creator score
  keywords   Show trending keywords (TF-IDF)
  topics     Phase 3 — NLP topic extraction from titles & comments
  sentiment  Phase 4 — Sentiment analysis on comments
  report     Generate full HTML report (+ terminal summary)

Examples
--------
  python main.py collect   --channel @mkbhd --pages 3
  python main.py analyze   --channel UCBcRF18a7Qf58cCRy5xuWwQ
  python main.py trending  --top 10
  python main.py channels  --top 5
  python main.py keywords  --top 20
  python main.py topics    --channel @mkbhd --n-topics 6
  python main.py sentiment --channel @mkbhd --source comments
  python main.py report    --channel @mkbhd
  python main.py report                        # all channels in DB
"""
import argparse
import sys
import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()


def print_banner():
    console.print(Panel.fit(
        "[bold cyan]📺 YouTube Analytics System[/bold cyan]\n"
        "[dim]Trend Scoring · Creator Intelligence · NLP Topics · Sentiment[/dim]",
        border_style="cyan",
        padding=(0, 2),
    ))


def cmd_collect(args):
    from collector import collect_channel
    if not args.channel:
        console.print("[red]--channel is required for collect[/red]")
        sys.exit(1)
    summary = collect_channel(
        args.channel,
        max_pages=args.pages,
        fetch_comments=not args.no_comments,
        comment_videos=args.comment_videos,
    )
    if summary:
        console.print()
        console.print(Panel(
            f"[bold white]Channel :[/bold white] {summary['title']}\n"
            f"[bold white]ID      :[/bold white] {summary['channel_id']}\n"
            f"[bold white]Subs    :[/bold white] {summary['subscribers']:,}\n"
            f"[bold white]Total V :[/bold white] {summary['total_views']:,}\n"
            f"[bold white]Videos  :[/bold white] {summary['video_count']} (fetched {summary['videos_fetched']})",
            title="[bold green]✓ Collection Complete[/bold green]",
            border_style="green",
        ))


def cmd_analyze(args):
    import db
    db.init_db()
    from report import print_creator_report
    if not args.channel:
        # Try all channels
        channels = db.get_all_channels()
        if not channels:
            console.print("[yellow]No channels in database. Run 'collect' first.[/yellow]")
            return
        for ch in channels:
            print_creator_report(ch["channel_id"])
    else:
        from collector import resolve_channel_id
        channel_id = resolve_channel_id(args.channel)
        if not channel_id:
            console.print(f"[red]Could not resolve: {args.channel}[/red]")
            sys.exit(1)
        print_creator_report(channel_id)


def cmd_trending(args):
    import db
    db.init_db()
    from report import print_trending_videos
    print_trending_videos(args.top)


def cmd_channels(args):
    import db
    db.init_db()
    from report import print_channel_leaderboard
    print_channel_leaderboard(args.top)


def cmd_keywords(args):
    import db
    db.init_db()
    from report import print_trending_keywords
    print_trending_keywords(args.top)


def cmd_topics(args):
    """Phase 3: NLP topic extraction."""
    import db
    db.init_db()
    from report import print_topics

    channel_id = None
    if args.channel:
        from collector import resolve_channel_id
        channel_id = resolve_channel_id(args.channel)
        if not channel_id:
            console.print(f"[red]Could not resolve: {args.channel}[/red]")
            import sys; sys.exit(1)

    print_topics(
        channel_id=channel_id,
        source=args.source,
        n_topics=args.n_topics,
    )


def cmd_sentiment(args):
    """Phase 4: Sentiment analysis on comments or titles."""
    import db
    db.init_db()
    from report import print_sentiment

    channel_id = None
    if args.channel:
        from collector import resolve_channel_id
        channel_id = resolve_channel_id(args.channel)
        if not channel_id:
            console.print(f"[red]Could not resolve: {args.channel}[/red]")
            import sys; sys.exit(1)

    print_sentiment(channel_id=channel_id, source=args.source)


def cmd_report(args):
    import db
    db.init_db()
    from report import (
        print_creator_report,
        print_trending_videos,
        print_channel_leaderboard,
        print_trending_keywords,
        print_topics,
        print_sentiment,
        generate_html_report,
    )

    channel_id = None
    if args.channel:
        from collector import resolve_channel_id
        channel_id = resolve_channel_id(args.channel)
        if not channel_id:
            console.print(f"[red]Could not resolve: {args.channel}[/red]")
            sys.exit(1)
        print_creator_report(channel_id)

    print_trending_videos(10)
    print_channel_leaderboard(10)
    print_trending_keywords(15)

    if not args.no_nlp:
        print_topics(channel_id=channel_id, source="titles", n_topics=6)
        print_sentiment(channel_id=channel_id, source="comments")

    console.print("[bold cyan]Generating HTML report…[/bold cyan]")
    path = generate_html_report(channel_id, include_nlp=not args.no_nlp)
    console.print(f"[bold green]✓ Report saved:[/bold green] {path}")

    if not args.no_open:
        try:
            subprocess.run(["open", str(path)], check=True)
        except Exception:
            pass


# ── Argument parser ────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ytanalysis",
        description="YouTube Analytics System — Trend Scoring & Creator Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # collect
    p_collect = sub.add_parser("collect", help="Fetch channel data from YouTube API")
    p_collect.add_argument("--channel", "-c", required=True,
        help="Channel ID (UC...) or @handle or custom URL slug")
    p_collect.add_argument("--pages", "-p", type=int, default=5,
        help="Number of video-list pages to fetch (50 vids/page, default: 5)")
    p_collect.add_argument("--no-comments", action="store_true",
        help="Skip fetching top comments")
    p_collect.add_argument("--comment-videos", type=int, default=10,
        help="Number of top videos to fetch comments for (default: 10)")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Creator intelligence report")
    p_analyze.add_argument("--channel", "-c", default=None,
        help="Channel ID or handle (omit = all channels in DB)")

    # trending
    p_trend = sub.add_parser("trending", help="Fastest growing videos by trend score")
    p_trend.add_argument("--top", "-n", type=int, default=10,
        help="Number of results (default: 10)")

    # channels
    p_ch = sub.add_parser("channels", help="Fastest growing channels leaderboard")
    p_ch.add_argument("--top", "-n", type=int, default=10,
        help="Number of results (default: 10)")

    # keywords
    p_kw = sub.add_parser("keywords", help="Trending keywords (TF-IDF)")
    p_kw.add_argument("--top", "-n", type=int, default=20,
        help="Number of keywords (default: 20)")

    # topics (Phase 3)
    p_topics = sub.add_parser("topics", help="Phase 3 — NLP topic extraction")
    p_topics.add_argument("--channel", "-c", default=None,
        help="Channel ID or handle (omit = all channels in DB)")
    p_topics.add_argument("--source", choices=["titles", "comments", "both"],
        default="both", help="Text source to cluster (default: both)")
    p_topics.add_argument("--n-topics", type=int, default=6,
        help="Number of topic clusters (default: 6)")

    # sentiment (Phase 4)
    p_sent = sub.add_parser("sentiment", help="Phase 4 — Sentiment analysis")
    p_sent.add_argument("--channel", "-c", default=None,
        help="Channel ID or handle (omit = all channels)")
    p_sent.add_argument("--source", choices=["comments", "titles"],
        default="comments", help="Text source to analyse (default: comments)")

    # report
    p_report = sub.add_parser("report", help="Full report — terminal + HTML")
    p_report.add_argument("--channel", "-c", default=None,
        help="Channel ID or handle (omit = all channels)")
    p_report.add_argument("--no-open", action="store_true",
        help="Do not auto-open the HTML file in browser")
    p_report.add_argument("--no-nlp", action="store_true",
        help="Skip Phase 3/4 NLP (faster, no model download)")

    return parser


def main():
    print_banner()
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "collect":   cmd_collect,
        "analyze":   cmd_analyze,
        "trending":  cmd_trending,
        "channels":  cmd_channels,
        "keywords":  cmd_keywords,
        "topics":    cmd_topics,
        "sentiment": cmd_sentiment,
        "report":    cmd_report,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
