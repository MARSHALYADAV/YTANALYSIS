"""
app.py — Flask web interface for YouTube Analytics System
=========================================================
Provides a browser-based dashboard for all 6 phases.

Routes
------
GET  /                    → Home page (channel input form)
POST /collect             → Trigger data collection for a channel
GET  /dashboard           → Full analytics dashboard
GET  /api/channel/<id>    → JSON: channel stats + creator score
GET  /api/trending        → JSON: top trending videos
GET  /api/keywords        → JSON: trending keywords
GET  /api/sentiment/<id>  → JSON: sentiment analysis result
GET  /api/insights/<id>   → JSON: Gemini AI insights
GET  /api/recommend/<id>  → JSON: content recommendations
GET  /report/<id>         → Full HTML report (rendered inline)
GET  /health              → Health check (for Render)
"""

import os
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import Flask, render_template_string, request, jsonify, redirect, url_for

import db
import analyzer
import config

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ── Background job tracker ─────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}   # job_id → {status, message, channel_id}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt(n):
    if n >= 1_000_000_000: return f"{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:     return f"{n/1_000_000:.1f}M"
    if n >= 1_000:         return f"{n/1_000:.1f}K"
    return str(n)

# ── Home Page ──────────────────────────────────────────────────────────────────

HOME_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>YouTube Analytics System</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0a0a0f;--surface:#12121a;--surface2:#1a1a26;--border:#2a2a40;
--text:#e8e8f0;--dim:#8888aa;--accent:#6366f1;--green:#22c55e;--red:#ef4444;--cyan:#06b6d4}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;
display:flex;flex-direction:column;align-items:center;justify-content:center;padding:2rem}
.hero{text-align:center;max-width:680px;width:100%}
.badge{display:inline-block;background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);
color:var(--accent);font-size:0.75rem;font-weight:600;padding:0.25rem 0.75rem;border-radius:999px;
letter-spacing:.08em;text-transform:uppercase;margin-bottom:1.5rem}
h1{font-size:3rem;font-weight:800;line-height:1.1;margin-bottom:1rem;
background:linear-gradient(135deg,#6366f1,#a855f7,#06b6d4);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.subtitle{color:var(--dim);font-size:1.05rem;margin-bottom:2.5rem;line-height:1.6}
.phases{display:flex;gap:0.5rem;flex-wrap:wrap;justify-content:center;margin-bottom:2.5rem}
.phase{background:var(--surface2);border:1px solid var(--border);border-radius:8px;
padding:0.4rem 0.9rem;font-size:0.78rem;color:var(--dim)}
.phase span{color:var(--text);font-weight:600}
.card{background:var(--surface);border:1px solid var(--border);border-radius:20px;
padding:2.5rem;width:100%;position:relative;overflow:hidden}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
background:linear-gradient(90deg,#6366f1,#a855f7,#06b6d4)}
.form-group{display:flex;gap:0.75rem;margin-bottom:1rem}
input[type=text]{flex:1;background:var(--surface2);border:1px solid var(--border);
border-radius:10px;padding:0.875rem 1.25rem;color:var(--text);font-family:'Inter',sans-serif;
font-size:0.95rem;outline:none;transition:border-color .2s}
input[type=text]:focus{border-color:var(--accent)}
input[type=text]::placeholder{color:var(--dim)}
.btn{background:linear-gradient(135deg,#6366f1,#a855f7);color:#fff;border:none;
border-radius:10px;padding:0.875rem 1.75rem;font-family:'Inter',sans-serif;font-size:0.95rem;
font-weight:600;cursor:pointer;transition:opacity .2s;white-space:nowrap}
.btn:hover{opacity:.85}
.btn-outline{background:transparent;border:1px solid var(--border);color:var(--dim)}
.options{display:flex;gap:1rem;align-items:center;flex-wrap:wrap}
select{background:var(--surface2);border:1px solid var(--border);border-radius:8px;
padding:0.5rem 1rem;color:var(--text);font-family:'Inter',sans-serif;font-size:0.85rem}
label{color:var(--dim);font-size:0.85rem}
.channels-list{margin-top:2rem}
.ch-item{display:flex;justify-content:space-between;align-items:center;padding:1rem;
background:var(--surface2);border-radius:10px;margin-bottom:0.5rem;transition:background .2s}
.ch-item:hover{background:rgba(99,102,241,0.1)}
.ch-name{font-weight:600}
.ch-meta{font-size:0.8rem;color:var(--dim)}
.ch-score{font-size:1.1rem;font-weight:700;color:var(--accent)}
.empty{color:var(--dim);text-align:center;padding:2rem;font-size:0.9rem}
.status-msg{padding:0.75rem 1rem;border-radius:8px;font-size:0.875rem;margin-top:1rem}
.status-ok{background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);color:var(--green)}
.status-err{background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);color:var(--red)}
footer{margin-top:3rem;color:var(--dim);font-size:0.8rem;text-align:center}
</style>
</head>
<body>
<div class="hero">
  <div class="badge">MCA Project · 6 Phases</div>
  <h1>YouTube Analytics System</h1>
  <p class="subtitle">Data Collection · Trend Scoring · NLP Topics · Sentiment · Gemini AI · Recommendations</p>
  <div class="phases">
    <div class="phase"><span>P1</span> YouTube API</div>
    <div class="phase"><span>P2</span> Trend Score</div>
    <div class="phase"><span>P3</span> NLP Topics</div>
    <div class="phase"><span>P4</span> Sentiment</div>
    <div class="phase"><span>P5</span> Gemini AI</div>
    <div class="phase"><span>P6</span> Recommend</div>
  </div>

  <div class="card">
    <form method="POST" action="/collect">
      <div class="form-group">
        <input type="text" name="channel" placeholder="@mkbhd  or  UCxxxxxx  or  channel URL" required>
        <button type="submit" class="btn">Analyse →</button>
      </div>
      <div class="options">
        <label>Pages (50 videos each):</label>
        <select name="pages">
          <option value="1">1 page (50 videos)</option>
          <option value="3" selected>3 pages (150 videos)</option>
          <option value="5">5 pages (250 videos)</option>
        </select>
        <label><input type="checkbox" name="comments" checked> Fetch comments</label>
      </div>
    </form>

    {% if msg %}
    <div class="status-msg {{ 'status-ok' if ok else 'status-err' }}">{{ msg }}</div>
    {% endif %}

    {% if channels %}
    <div class="channels-list">
      <p style="color:var(--dim);font-size:0.8rem;margin-bottom:0.75rem;text-transform:uppercase;letter-spacing:.08em">Analysed Channels</p>
      {% for ch in channels %}
      <a href="/dashboard?channel={{ ch.channel_id }}" style="text-decoration:none">
        <div class="ch-item">
          <div>
            <div class="ch-name">{{ ch.title }}</div>
            <div class="ch-meta">{{ '{:,}'.format(ch.subscriber_count) }} subscribers · {{ '{:,}'.format(ch.video_count) }} videos</div>
          </div>
          <div class="ch-score">{{ ch.creator_score }}<span style="font-size:0.65rem;color:var(--dim)">/100</span></div>
        </div>
      </a>
      {% endfor %}
    </div>
    {% else %}
    <div class="empty">No channels analysed yet. Enter a channel above to get started.</div>
    {% endif %}
  </div>
</div>
<footer>YouTube Analytics System · Built with Python + Gemini AI · MCA Project 2024</footer>
</body>
</html>"""


@app.route("/")
def home():
    db.init_db()
    channels_raw = db.get_all_channels()
    channels = []
    for ch in channels_raw:
        score = analyzer.compute_creator_score(ch["channel_id"])
        channels.append({
            "channel_id":      ch["channel_id"],
            "title":           ch["title"],
            "subscriber_count":ch["subscriber_count"],
            "video_count":     ch["video_count"],
            "creator_score":   score.get("creator_score", 0),
        })
    msg = request.args.get("msg", "")
    ok  = request.args.get("ok", "1") == "1"
    return render_template_string(HOME_HTML, channels=channels, msg=msg, ok=ok)


# ── Collect ────────────────────────────────────────────────────────────────────

@app.route("/collect", methods=["POST"])
def collect():
    channel    = request.form.get("channel", "").strip()
    pages      = int(request.form.get("pages", 3))
    comments   = "comments" in request.form

    if not channel:
        return redirect(url_for("home", msg="Please enter a channel identifier.", ok=0))

    try:
        from collector import collect_channel
        summary = collect_channel(channel, max_pages=pages, fetch_comments=comments)
        if not summary:
            return redirect(url_for("home", msg=f"Channel not found: {channel}", ok=0))
        msg = f"✅ Collected {summary['videos_fetched']} videos for {summary['title']}"
        return redirect(url_for("dashboard", channel=summary["channel_id"], msg=msg))
    except Exception as e:
        return redirect(url_for("home", msg=f"Error: {e}", ok=0))


# ── Dashboard ──────────────────────────────────────────────────────────────────

@app.route("/dashboard")
def dashboard():
    db.init_db()
    channel_id = request.args.get("channel")
    msg        = request.args.get("msg", "")

    if not channel_id:
        channels = db.get_all_channels()
        if not channels:
            return redirect(url_for("home", msg="No channels found. Collect data first."))
        channel_id = channels[0]["channel_id"]

    # Redirect to home with a clear error if the channel isn't in database
    ch = db.get_channel(channel_id)
    if not ch:
        return redirect(url_for("home", msg=f"Channel {channel_id} not found in database. Please run collection first.", ok=0))

    from report import generate_html_report
    try:
        html_path = generate_html_report(channel_id, include_nlp=False)
        return html_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"<pre>Error generating report: {e}</pre>", 500


# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.route("/api/channel/<channel_id>")
def api_channel(channel_id):
    db.init_db()
    ch = db.get_channel(channel_id)
    if not ch:
        return jsonify({"error": "Channel not found"}), 404
    score = analyzer.compute_creator_score(channel_id)
    return jsonify({
        "channel_id":       ch["channel_id"],
        "title":            ch["title"],
        "subscriber_count": ch["subscriber_count"],
        "view_count":       ch["view_count"],
        "video_count":      ch["video_count"],
        "fetched_at":       ch["fetched_at"],
        **score,
    })


@app.route("/api/trending")
def api_trending():
    db.init_db()
    top_n = int(request.args.get("n", 10))
    return jsonify(analyzer.fastest_growing_videos(top_n))


@app.route("/api/keywords")
def api_keywords():
    db.init_db()
    top_n = int(request.args.get("n", 20))
    kws = analyzer.trending_keywords(top_n)
    return jsonify([{"keyword": k, "score": round(s, 2)} for k, s in kws])


@app.route("/api/channels")
def api_channels():
    db.init_db()
    top_n = int(request.args.get("n", 10))
    return jsonify(analyzer.fastest_growing_channels(top_n))


@app.route("/api/sentiment/<channel_id>")
def api_sentiment(channel_id):
    db.init_db()
    from sentiment import analyze_comments
    result = analyze_comments(channel_id)
    return jsonify(result)


@app.route("/api/insights/<channel_id>")
def api_insights(channel_id):
    db.init_db()
    from gemini_insight import generate_channel_insights
    result = generate_channel_insights(channel_id)
    if "error" in result:
        return jsonify(result), 400
    # Return only the structured sections (not the raw prompt)
    return jsonify({
        "channel_title": result["channel_title"],
        "sections":      result["sections"],
    })


@app.route("/api/recommend/<channel_id>")
def api_recommend(channel_id):
    db.init_db()
    from recommender import quick_recommend_from_keywords
    kws  = analyzer.trending_keywords(10)
    recs = quick_recommend_from_keywords(channel_id, kws)
    return jsonify(recs)


@app.route("/report/<channel_id>")
def full_report(channel_id):
    db.init_db()
    # Redirect to home with a clear error if the channel isn't in database
    ch = db.get_channel(channel_id)
    if not ch:
        return redirect(url_for("home", msg=f"Channel {channel_id} not found in database. Please run collection first.", ok=0))

    from report import generate_html_report
    try:
        path = generate_html_report(channel_id, include_nlp=False)
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"<pre>Error: {e}</pre>", 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    db.init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
