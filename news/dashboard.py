from __future__ import annotations

import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

# Local imports
try:
    from news.sentiment import add_vader_sentiment
    from news.main import run_once_from_dashboard
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from news.sentiment import add_vader_sentiment
    from news.main import run_once_from_dashboard


DATA_LATEST = Path("data/latest_headlines.csv")
RUNS_DIR = Path("data/runs")

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "as", "at", "by",
    "from", "is", "are", "was", "were", "be", "been", "it", "that", "this", "these", "those",
    "after", "before", "over", "under", "into", "about", "amid", "says", "say", "new", "latest",
    "today", "live", "update", "breaking",
}


# -------------------------
# Styling 
# -------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* Global */
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        [data-testid="stSidebar"] { padding-top: 0.75rem; }

        /* Header */
        .ai-header {
          background: linear-gradient(90deg, rgba(76, 110, 245, 0.18), rgba(32, 201, 151, 0.12));
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 18px;
          padding: 18px 18px 14px 18px;
          margin-bottom: 14px;
        }
        .ai-title {
          font-size: 1.65rem;
          font-weight: 800;
          line-height: 1.2;
          margin: 0;
        }
        .ai-subtitle {
          opacity: 0.85;
          margin-top: 6px;
          font-size: 0.95rem;
        }

        /* KPI Cards */
        .kpi-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 12px;
          margin-top: 10px;
          margin-bottom: 12px;
        }
        .kpi {
          border-radius: 16px;
          padding: 14px 14px 12px 14px;
          border: 1px solid rgba(255,255,255,0.10);
          background: rgba(255,255,255,0.03);
        }
        .kpi-label {
          font-size: 0.82rem;
          opacity: 0.75;
          margin-bottom: 6px;
        }
        .kpi-value {
          font-size: 1.35rem;
          font-weight: 800;
          margin: 0;
        }
        .kpi-hint {
          font-size: 0.8rem;
          opacity: 0.70;
          margin-top: 6px;
        }

        /* Section headings */
        .section-title {
          font-size: 1.05rem;
          font-weight: 800;
          margin-top: 0.25rem;
          margin-bottom: 0.25rem;
        }
        .section-caption {
          opacity: 0.80;
          font-size: 0.9rem;
          margin-bottom: 0.6rem;
        }

        /* Small badge */
        .badge {
          display: inline-block;
          padding: 3px 10px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.04);
          font-size: 0.82rem;
          opacity: 0.9;
        }

        /* Responsive */
        @media (max-width: 1000px) {
          .kpi-grid { grid-template-columns: repeat(2, 1fr); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -------------------------
# Data
# -------------------------
def load_latest() -> pd.DataFrame:
    if not DATA_LATEST.exists():
        return pd.DataFrame()

    df = pd.read_csv(DATA_LATEST)

    # Parse datetime
    if "published" in df.columns:
        df["published"] = pd.to_datetime(df["published"], errors="coerce")

    # Ensure columns exist
    for col in ["title", "source", "summary", "link", "category", "relevance"]:
        if col not in df.columns:
            df[col] = ""

    # Add sentiment if missing
    if "sentiment_compound" not in df.columns or "sentiment_label" not in df.columns:
        df = add_vader_sentiment(df)

    return df


def list_last_snapshots(limit: int = 14) -> List[Path]:
    if not RUNS_DIR.exists():
        return []
    files = sorted(RUNS_DIR.glob("headlines_*.csv"), reverse=True)
    return files[:limit]


def tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]


def keyword_counts(df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["keyword", "count"])

    blob = (df["title"].fillna("") + " " + df["summary"].fillna("")).astype(str)
    c = Counter()
    for s in blob.tolist():
        c.update(tokenize(s))

    common = c.most_common(top_k)
    return pd.DataFrame(common, columns=["keyword", "count"])


def trending_today_vs_last_runs(today_df: pd.DataFrame, snapshot_paths: List[Path], top_k: int = 15) -> pd.DataFrame:
    """
    Trending = today_count - avg(previous runs)
    """
    today_map = keyword_counts(today_df, top_k=250).set_index("keyword")["count"].to_dict()

    prev_total = Counter()
    prev_n = 0
    for p in snapshot_paths:
        try:
            d = pd.read_csv(p)
            for col in ["title", "summary"]:
                if col not in d.columns:
                    d[col] = ""
            tmp = keyword_counts(d, top_k=300)
            prev_total.update(dict(zip(tmp["keyword"], tmp["count"])))
            prev_n += 1
        except Exception:
            continue

    prev_n = max(prev_n, 1)

    candidates = set(today_map.keys())
    for kw, _ in prev_total.most_common(60):
        candidates.add(kw)

    rows = []
    for kw in candidates:
        today = float(today_map.get(kw, 0))
        prev_avg = float(prev_total.get(kw, 0)) / prev_n
        rows.append((kw, today, prev_avg, today - prev_avg))

    out = pd.DataFrame(rows, columns=["keyword", "today_count", "prev_avg", "delta"])
    out = out.sort_values(["delta", "today_count"], ascending=[False, False]).head(top_k).reset_index(drop=True)
    return out


def safe_pct(num: float) -> str:
    return f"{(num * 100):.1f}%"


# -------------------------
# UI
# -------------------------
def app():
    st.set_page_config(page_title="Smart News • AI Analytics", layout="wide")
    inject_css()

    # Sidebar: controls
    st.sidebar.markdown("### Controls")
    refresh_min = st.sidebar.number_input("Auto-reload (minutes)", min_value=0, max_value=120, value=5, step=1)
    if refresh_min > 0 and hasattr(st, "autorefresh"):
        st.autorefresh(interval=int(refresh_min * 60 * 1000), key="ai_autorefresh")

    kw_refresh = st.sidebar.text_input("Refresh keywords (optional)", value="").strip()
    kw_list = [k for k in kw_refresh.split() if k] if kw_refresh else None

    if st.sidebar.button("🔄 Refresh News Now", use_container_width=True):
        with st.spinner("Fetching & processing news…"):
            try:
                run_once_from_dashboard(keywords=kw_list)
                st.sidebar.success("Updated.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Refresh failed: {e}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")

    df = load_latest()
    if df.empty:
        st.markdown(
            """
            <div class="ai-header">
              <div class="ai-title">🗞️ Smart News • AI Analytics</div>
              <div class="ai-subtitle">No data yet. Click <b>Refresh News Now</b> in the sidebar.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    # Build filter options
    categories = sorted(df["category"].fillna("General").unique().tolist())
    sources = sorted(df["source"].fillna("Unknown").unique().tolist())
    sentiments = ["Positive", "Neutral", "Negative"]

    sel_cats = st.sidebar.multiselect("Category", options=categories, default=categories)
    sel_src = st.sidebar.multiselect("Source", options=sources, default=sources)
    sel_sent = st.sidebar.multiselect("Sentiment", options=sentiments, default=sentiments)
    q = st.sidebar.text_input("Search (title + summary)", value="").strip().lower()

    sort_mode = st.sidebar.selectbox("Sort", ["Relevance", "Most recent"], index=0)

    # Apply filters
    filtered = df.copy()
    if sel_cats:
        filtered = filtered[filtered["category"].fillna("General").isin(sel_cats)]
    if sel_src:
        filtered = filtered[filtered["source"].fillna("Unknown").isin(sel_src)]
    if sel_sent:
        filtered = filtered[filtered["sentiment_label"].isin(sel_sent)]
    if q:
        blob = (filtered["title"].fillna("") + " " + filtered["summary"].fillna("")).str.lower()
        filtered = filtered[blob.str.contains(q, na=False)]

    # Sort
    if sort_mode == "Most recent":
        filtered = filtered.sort_values("published", ascending=False)
    else:
        if "relevance" in filtered.columns:
            filtered = filtered.sort_values(["relevance", "published"], ascending=[False, False])
        else:
            filtered = filtered.sort_values("published", ascending=False)

    # Header
    latest_ts = filtered["published"].max()
    latest_txt = str(latest_ts) if pd.notna(latest_ts) else "N/A"
    st.markdown(
        f"""
        <div class="ai-header">
          <div class="ai-title">🗞️ Smart News • AI Analytics</div>
          <div class="ai-subtitle">
            Signal-driven overview of today’s headlines • <span class="badge">Latest: {latest_txt}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI cards
    pos_rate = (filtered["sentiment_label"] == "Positive").mean() if len(filtered) else 0.0
    neg_rate = (filtered["sentiment_label"] == "Negative").mean() if len(filtered) else 0.0
    avg_sent = filtered["sentiment_compound"].mean() if len(filtered) else 0.0
    avg_rel = filtered["relevance"].mean() if ("relevance" in filtered.columns and len(filtered)) else 0.0

    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi">
            <div class="kpi-label">Headlines</div>
            <div class="kpi-value">{len(filtered)}</div>
            <div class="kpi-hint">After filters + dedupe</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Sources</div>
            <div class="kpi-value">{filtered["source"].nunique()}</div>
            <div class="kpi-hint">Distinct publishers</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Avg Sentiment</div>
            <div class="kpi-value">{avg_sent:.2f}</div>
            <div class="kpi-hint">Positive {safe_pct(pos_rate)} • Negative {safe_pct(neg_rate)}</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Avg Relevance</div>
            <div class="kpi-value">{avg_rel:.2f}</div>
            <div class="kpi-hint">Keyword + recency score</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tabs: Overview / Trends / Sentiment / Explorer
    tab_overview, tab_trends, tab_sent, tab_explorer = st.tabs(
        ["📌 Overview", "📈 Trends", "😊 Sentiment", "🧭 Explorer"]
    )

    with tab_overview:
        st.markdown('<div class="section-title">Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Where the attention is today.</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            cat_counts = filtered["category"].fillna("General").value_counts().head(12)
            st.bar_chart(cat_counts)
        with c2:
            src_counts = filtered["source"].fillna("Unknown").value_counts().head(12)
            st.bar_chart(src_counts)

        st.markdown("---")
        st.markdown('<div class="section-title">Top Headlines</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Highest signal items after filtering.</div>', unsafe_allow_html=True)

        topn = filtered.head(15).copy()
        topn["open"] = topn["link"].apply(lambda u: f"[Open]({u})" if isinstance(u, str) and u.startswith("http") else "")
        show_cols = ["published", "category", "source", "sentiment_label", "sentiment_compound", "title", "open"]
        for col in show_cols:
            if col not in topn.columns:
                topn[col] = ""
        st.dataframe(topn[show_cols], use_container_width=True, hide_index=True)

    with tab_trends:
        st.markdown('<div class="section-title">Trending Keywords</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Today vs average of the last 7 runs (delta = trending up).</div>',
            unsafe_allow_html=True,
        )

        snapshots = list_last_snapshots(limit=7)
        trend = trending_today_vs_last_runs(df, snapshots, top_k=20)
        st.dataframe(trend, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Top Keywords Today</div>', unsafe_allow_html=True)
        kw = keyword_counts(filtered, top_k=20)
        if not kw.empty:
            st.bar_chart(kw.set_index("keyword")["count"])
        else:
            st.info("No keywords to display (empty dataset after filters).")

    with tab_sent:
        st.markdown('<div class="section-title">Sentiment Overview</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">VADER sentiment on title + summary.</div>',
            unsafe_allow_html=True,
        )

        s1, s2 = st.columns(2)
        with s1:
            dist = filtered["sentiment_label"].value_counts()
            st.bar_chart(dist)

        with s2:
            # Altair-safe histogram (no IntervalIndex)
            bins = pd.cut(filtered["sentiment_compound"], bins=[-1, -0.6, -0.2, 0.2, 0.6, 1], include_lowest=True)
            hist = bins.value_counts().sort_index().reset_index()
            hist.columns = ["range", "count"]
            hist["range"] = hist["range"].astype(str)
            st.bar_chart(hist.set_index("range")["count"])

        st.markdown("---")
        st.markdown('<div class="section-title">Most Negative / Most Positive</div>', unsafe_allow_html=True)

        a, b = st.columns(2)
        with a:
            neg = filtered.sort_values("sentiment_compound", ascending=True).head(10).copy()
            neg["open"] = neg["link"].apply(lambda u: f"[Open]({u})" if isinstance(u, str) and u.startswith("http") else "")
            st.caption("Most Negative")
            st.dataframe(
                neg[["sentiment_compound", "sentiment_label", "source", "title", "open"]],
                use_container_width=True,
                hide_index=True,
            )
        with b:
            pos = filtered.sort_values("sentiment_compound", ascending=False).head(10).copy()
            pos["open"] = pos["link"].apply(lambda u: f"[Open]({u})" if isinstance(u, str) and u.startswith("http") else "")
            st.caption("Most Positive")
            st.dataframe(
                pos[["sentiment_compound", "sentiment_label", "source", "title", "open"]],
                use_container_width=True,
                hide_index=True,
            )

    with tab_explorer:
        st.markdown('<div class="section-title">Headline Explorer</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Open, skim summaries, and quickly understand what matters.</div>',
            unsafe_allow_html=True,
        )

        n = st.slider("Rows", min_value=10, max_value=200, value=60, step=10)
        view = filtered.head(n).copy()
        view["open"] = view["link"].apply(lambda u: f"[Open]({u})" if isinstance(u, str) and u.startswith("http") else "")

        cols = ["published", "category", "source", "sentiment_label", "sentiment_compound", "title", "open"]
        for col in cols:
            if col not in view.columns:
                view[col] = ""

        st.dataframe(view[cols], use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Quick Reader</div>', unsafe_allow_html=True)

        for _, r in view.head(20).iterrows():
            title = str(r.get("title", ""))
            source = str(r.get("source", ""))
            with st.expander(f"{title} — {source}"):
                st.write(f"**Category:** {r.get('category', '')}")
                st.write(f"**Sentiment:** {r.get('sentiment_label', '')} ({float(r.get('sentiment_compound', 0.0)):.2f})")
                st.write(f"**Published:** {r.get('published', '')}")
                link = str(r.get("link", ""))
                if link:
                    st.markdown(f"**Link:** {link}")
                summary = str(r.get("summary", ""))
                if summary:
                    st.write(summary)


if __name__ == "__main__":
    app()