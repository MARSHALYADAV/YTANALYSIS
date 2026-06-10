"""
collector.py — YouTube Data API v3 collection layer.

Fetches channel, video, and comment data, then persists to SQLite.

Usage (called from main.py):
    from collector import collect_channel
    collect_channel("UCxxxxxx", max_pages=5, fetch_comments=True)
"""
import json
import time
from typing import Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

import config
import db

console = Console()

# ── Build YouTube API client ───────────────────────────────────────────────────

def _build_youtube():
    return build(
        config.YOUTUBE_API_SERVICE,
        config.YOUTUBE_API_VERSION,
        developerKey=config.YOUTUBE_API_KEY,
        cache_discovery=False,
    )


# ── Channel resolution ─────────────────────────────────────────────────────────

def resolve_channel_id(identifier: str) -> Optional[str]:
    """
    Accepts:
      - Raw channel ID (starts with UC)
      - @handle  (e.g. @mkbhd)
      - Custom URL slug (e.g. mkbhd)
    Returns the channel_id string or None.
    """
    yt = _build_youtube()

    # Already a channel ID
    if identifier.startswith("UC") and len(identifier) == 24:
        return identifier

    # Handle @handle or plain slug
    handle = identifier.lstrip("@")

    try:
        resp = yt.search().list(
            part="snippet",
            q=handle,
            type="channel",
            maxResults=1,
        ).execute()
        items = resp.get("items", [])
        if items:
            return items[0]["snippet"]["channelId"]
    except HttpError as e:
        console.print(f"[red]API error resolving channel: {e}[/red]")
    return None


# ── Fetch channel details ─────────────────────────────────────────────────────

def fetch_channel_details(channel_id: str) -> Optional[dict]:
    """Fetch and store channel-level metadata."""
    yt = _build_youtube()
    try:
        resp = yt.channels().list(
            part="snippet,statistics,contentDetails",
            id=channel_id,
        ).execute()
    except HttpError as e:
        console.print(f"[red]API error fetching channel: {e}[/red]")
        return None

    items = resp.get("items", [])
    if not items:
        console.print(f"[red]Channel not found: {channel_id}[/red]")
        return None

    item = items[0]
    snippet = item.get("snippet", {})
    stats   = item.get("statistics", {})
    thumb   = snippet.get("thumbnails", {}).get("high", {}).get("url", "")

    data = {
        "channel_id":       channel_id,
        "title":            snippet.get("title", ""),
        "description":      snippet.get("description", "")[:1000],
        "custom_url":       snippet.get("customUrl", ""),
        "country":          snippet.get("country", ""),
        "subscriber_count": int(stats.get("subscriberCount", 0)),
        "view_count":       int(stats.get("viewCount", 0)),
        "video_count":      int(stats.get("videoCount", 0)),
        "published_at":     snippet.get("publishedAt", ""),
        "thumbnail_url":    thumb,
    }
    db.upsert_channel(data)
    return data


# ── Fetch video list ──────────────────────────────────────────────────────────

def fetch_video_list(channel_id: str, max_pages: int = 5) -> list[str]:
    """
    Return a list of video_ids from the channel's uploads playlist.
    max_pages × 50 = max videos fetched.
    """
    yt     = _build_youtube()
    # Get uploads playlist ID
    try:
        resp = yt.channels().list(
            part="contentDetails",
            id=channel_id,
        ).execute()
    except HttpError as e:
        console.print(f"[red]API error fetching uploads playlist: {e}[/red]")
        return []

    items = resp.get("items", [])
    if not items:
        return []

    uploads_playlist = (
        items[0]
        .get("contentDetails", {})
        .get("relatedPlaylists", {})
        .get("uploads", "")
    )
    if not uploads_playlist:
        return []

    video_ids: list[str] = []
    page_token: Optional[str] = None

    for page in range(max_pages):
        try:
            kwargs = dict(
                part="snippet",
                playlistId=uploads_playlist,
                maxResults=50,
            )
            if page_token:
                kwargs["pageToken"] = page_token

            resp = yt.playlistItems().list(**kwargs).execute()
        except HttpError as e:
            console.print(f"[red]API error on page {page+1}: {e}[/red]")
            break

        for item in resp.get("items", []):
            vid = item["snippet"]["resourceId"].get("videoId")
            if vid:
                video_ids.append(vid)

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return video_ids


# ── Fetch video details (batched) ──────────────────────────────────────────────

def fetch_video_details(video_ids: list[str], channel_id: str) -> list[dict]:
    """
    Fetch full stats for up to N video IDs in batches of 50.
    Stores results in DB and returns the list of dicts.
    """
    yt      = _build_youtube()
    results: list[dict] = []

    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        try:
            resp = yt.videos().list(
                part="snippet,statistics,contentDetails",
                id=",".join(batch),
            ).execute()
        except HttpError as e:
            console.print(f"[red]API error fetching video batch: {e}[/red]")
            continue

        rows: list[dict] = []
        for item in resp.get("items", []):
            snippet = item.get("snippet", {})
            stats   = item.get("statistics", {})
            content = item.get("contentDetails", {})
            tags    = snippet.get("tags", [])
            thumb   = snippet.get("thumbnails", {}).get("high", {}).get("url", "")

            row = {
                "video_id":     item["id"],
                "channel_id":   channel_id,
                "title":        snippet.get("title", ""),
                "description":  snippet.get("description", "")[:500],
                "published_at": snippet.get("publishedAt", ""),
                "duration":     content.get("duration", ""),
                "view_count":   int(stats.get("viewCount", 0)),
                "like_count":   int(stats.get("likeCount", 0)),
                "comment_count":int(stats.get("commentCount", 0)),
                "thumbnail_url":thumb,
                "tags":         json.dumps(tags[:20]),  # store up to 20 tags
            }
            rows.append(row)
            results.append(row)

        db.upsert_videos(rows)
        time.sleep(0.1)  # gentle rate limiting

    return results


# ── Fetch top comments for a video ────────────────────────────────────────────

def fetch_top_comments(video_id: str, max_results: int = 20) -> list[dict]:
    """Fetch top-level comments sorted by relevance. Returns comment rows."""
    yt = _build_youtube()
    rows: list[dict] = []
    try:
        resp = yt.commentThreads().list(
            part="snippet",
            videoId=video_id,
            order="relevance",
            maxResults=min(max_results, 100),
            textFormat="plainText",
        ).execute()
    except HttpError as e:
        # Comments disabled on some videos — not an error
        if "commentsDisabled" in str(e) or "403" in str(e):
            return []
        console.print(f"[yellow]Comments unavailable for {video_id}: {e}[/yellow]")
        return []

    for item in resp.get("items", []):
        top = item["snippet"]["topLevelComment"]["snippet"]
        rows.append({
            "comment_id":  item["id"],
            "video_id":    video_id,
            "author":      top.get("authorDisplayName", ""),
            "text":        top.get("textDisplay", "")[:500],
            "like_count":  int(top.get("likeCount", 0)),
            "published_at":top.get("publishedAt", ""),
        })

    if rows:
        db.upsert_comments(rows)
    return rows


# ── Master collection function ─────────────────────────────────────────────────

def collect_channel(
    identifier: str,
    max_pages: int = config.DEFAULT_MAX_PAGES,
    fetch_comments: bool = True,
    comment_videos: int = 10,
) -> dict:
    """
    Full pipeline for one channel:
      1. Resolve channel ID
      2. Fetch channel details
      3. Fetch video list
      4. Fetch video details (batched)
      5. Fetch top comments for top N videos (by view count)

    Returns a summary dict.
    """
    db.init_db()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:

        # ── Step 1: Resolve channel ─────────────────────────────────────────
        t = progress.add_task("Resolving channel ID…", total=None)
        channel_id = resolve_channel_id(identifier)
        if not channel_id:
            console.print(f"[red]Could not resolve channel: {identifier}[/red]")
            return {}
        progress.update(t, description=f"✓ Channel ID: {channel_id}", completed=1, total=1)

        # ── Step 2: Channel details ─────────────────────────────────────────
        t2 = progress.add_task("Fetching channel details…", total=None)
        channel_data = fetch_channel_details(channel_id)
        if not channel_data:
            return {}
        title = channel_data["title"]
        progress.update(t2, description=f"✓ Channel: {title}", completed=1, total=1)

        # ── Step 3: Video list ──────────────────────────────────────────────
        t3 = progress.add_task(f"Fetching video list ({max_pages} pages)…", total=None)
        video_ids = fetch_video_list(channel_id, max_pages=max_pages)
        progress.update(t3, description=f"✓ Found {len(video_ids)} videos", completed=1, total=1)

        # ── Step 4: Video details ───────────────────────────────────────────
        t4 = progress.add_task("Fetching video details…", total=len(video_ids))
        videos = []
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i+50]
            chunk = fetch_video_details(batch, channel_id)
            videos.extend(chunk)
            progress.update(t4, advance=len(batch))

        # ── Step 5: Top comments ────────────────────────────────────────────
        if fetch_comments and videos:
            top_vids = sorted(videos, key=lambda v: v["view_count"], reverse=True)
            top_vids = top_vids[:comment_videos]
            t5 = progress.add_task(
                f"Fetching comments for top {len(top_vids)} videos…",
                total=len(top_vids),
            )
            for v in top_vids:
                fetch_top_comments(v["video_id"], max_results=config.DEFAULT_TOP_COMMENTS)
                progress.update(t5, advance=1)

    summary = {
        "channel_id":       channel_id,
        "title":            title,
        "subscribers":      channel_data["subscriber_count"],
        "total_views":      channel_data["view_count"],
        "video_count":      channel_data["video_count"],
        "videos_fetched":   len(videos),
    }
    return summary
