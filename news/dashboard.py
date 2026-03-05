from __future__ import annotations

import re
import sys
from collections import Counter
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Import internal modules
try:
    from news.storage import NewsStore
    from news.main import run_once_from_dashboard
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from news.storage import NewsStore
    from news.main import run_once_from_dashboard


DB_PATH = "data/news.db"

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
        .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
        [data-testid="stSidebar"] { padding-top: 0.65rem; }

        .ai-header {
          background: linear-gradient(90deg, rgba(76,110,245,0.18), rgba(32,201,151,0.12));
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 18px;
          padding: 18px 18px 14px 18px;
          margin-bottom: 14px;
        }
        .ai-title { font-size: 1.55rem; font-weight: 900; margin: 0; line-height: 1.2; }
        .ai-subtitle { opacity: 0.85; margin-top: 7px; font-size: 0.95rem; }

        .badge {
          display: inline-block;
          padding: 3px 10px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.04);
          font-size: 0.82rem;
          opacity: 0.95;
        }

        .kpi-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 12px;
          margin-top: 8px;
          margin-bottom: 12px;
        }
        .kpi {
          border-radius: 16px;
          padding: 14px 14px 12px 14px;
          border: 1px solid rgba(255,255,255,0.10);
          background: rgba(255,255,255,0.03);
        }
        .kpi-label { font-size: 0.82rem; opacity: 0.75; margin-bottom: 6px; }
        .kpi-value { font-size: 1.35rem; font-weight: 900; margin: 0; }
        .kpi-hint { font-size: 0.8rem; opacity: 0.70; margin-top: 6px; }

        .section-title { font-size: 1.05rem; font-weight: 900; margin-top: 0.25rem; margin-bottom: 0.25rem; }
        .section-caption { opacity: 0.80; font-size: 0.9rem; margin-bottom: 0.6rem; }

        @media (max-width: 1000px) {
          .kpi-grid { grid-template-columns: repeat(2, 1fr); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_pct(num: float) -> str:
    return f"{(num * 100):.1f}%"


# -------------------------
# Keyword utilities
# -------------------------
def tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]


def keyword_counts_from_texts(texts: List[str], top_k: int = 20) -> Dict[str, int]:
    c = Counter()
    for s in texts:
        c.update(tokenize(s))
    return dict(c.most_common(top_k))


def df_keyword_counts(df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["keyword", "count"])
    blob = (df["title"].fillna("") + " " + df["summary"].fillna("")).astype(str).tolist()
    c = Counter()
    for s in blob:
        c.update(tokenize(s))
    out = pd.DataFrame(c.most_common(top_k), columns=["keyword", "count"])
    return out


def trending_today_vs_last7days(store: NewsStore, today: date, top_k: int = 15) -> pd.DataFrame:
    """
    Trending keywords: today_count - avg(prev 6 days)
    Uses unique dates (not runs).
    """
    # We store published_at in DB. We will query by published_at range and then group.
    start = (today - timedelta(days=6)).isoformat()
    end = (today + timedelta(days=1)).isoformat()

    df7 = store.query_headlines(
        since_iso=start,
        until_iso=end,
        limit=5000,
        order_by="published_at DESC",
    )

    if df7.empty or "published_at" not in df7.columns:
        return pd.DataFrame(columns=["keyword", "today_count", "prev_avg", "delta"])

    df7["day"] = pd.to_datetime(df7["published_at"], errors="coerce").dt.date
    df_today = df7[df7["day"] == today].copy()
    df_prev = df7[df7["day"] != today].copy()

    today_map = df_keyword_counts(df_today, top_k=250).set_index("keyword")["count"].to_dict() if not df_today.empty else {}

    # average across previous unique days
    prev_days = sorted(df_prev["day"].dropna().unique().tolist())
    prev_day_counts: Counter = Counter()

    if prev_days:
        for d in prev_days:
            ddf = df_prev[df_prev["day"] == d]
            tmp = df_keyword_counts(ddf, top_k=300)
            prev_day_counts.update(dict(zip(tmp["keyword"], tmp["count"])))
        prev_avg_div = max(len(prev_days), 1)
    else:
        prev_avg_div = 1

    # candidate keywords = top today + some from previous
    candidates = set(today_map.keys())
    for kw, _ in prev_day_counts.most_common(60):
        candidates.add(kw)

    rows = []
    for kw in candidates:
        today_count = float(today_map.get(kw, 0))
        prev_avg = float(prev_day_counts.get(kw, 0)) / prev_avg_div
        rows.append((kw, today_count, prev_avg, today_count - prev_avg))

    out = pd.DataFrame(rows, columns=["keyword", "today_count", "prev_avg", "delta"])
    out = out.sort_values(["delta", "today_count"], ascending=[False, False]).head(top_k).reset_index(drop=True)
    return out


# -------------------------
# Main app
# -------------------------
def app():
    st.set_page_config(page_title="Smart News • AI Analytics", layout="wide")
    inject_css()

    store = NewsStore(DB_PATH)

    # ---------- Sidebar: controls ----------
    st.sidebar.markdown("### Controls")
    refresh_min = st.sidebar.number_input("Auto-reload (minutes)", min_value=0, max_value=120, value=5, step=1)
    if refresh_min > 0 and hasattr(st, "autorefresh"):
        st.autorefresh(interval=int(refresh_min * 60 * 1000), key="auto_reload")

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

    unseen_only = st.sidebar.toggle("Only unseen", value=True)

    # pull a broad dataset (we'll filter in UI too)
    base_df = store.query_headlines(unseen_only=unseen_only, limit=5000, order_by="published_at DESC")

    if base_df.empty:
        st.markdown(
            """
            <div class="ai-header">
              <div class="ai-title">🗞️ Smart News • AI Analytics</div>
              <div class="ai-subtitle">No headlines in SQLite yet. Click <b>Refresh News Now</b> to ingest.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    # Normalize / ensure datetime columns
    if "published_at" in base_df.columns:
        base_df["published_at"] = pd.to_datetime(base_df["published_at"], errors="coerce")

    # Date range filter
    min_dt = base_df["published_at"].min()
    max_dt = base_df["published_at"].max()
    if pd.isna(min_dt) or pd.isna(max_dt):
        start_date = st.sidebar.date_input("Start date", value=date.today())
        end_date = st.sidebar.date_input("End date", value=date.today())
    else:
        start_date = st.sidebar.date_input("Start date", value=min_dt.date())
        end_date = st.sidebar.date_input("End date", value=max_dt.date())

    if start_date > end_date:
        st.sidebar.error("Start date must be ≤ end date.")
        st.stop()

    # Category/source/sentiment filters
    categories = sorted(base_df["category"].fillna("General").unique().tolist())
    sources = sorted(base_df["source"].fillna("Unknown").unique().tolist())
    sentiments = ["Positive", "Neutral", "Negative"]

    sel_cats = st.sidebar.multiselect("Category", options=categories, default=categories)
    sel_src = st.sidebar.multiselect("Source", options=sources, default=sources)
    sel_sent = st.sidebar.multiselect("Sentiment", options=sentiments, default=sentiments)
    q = st.sidebar.text_input("Search (title + summary)", value="").strip().lower()

    sort_mode = st.sidebar.selectbox("Sort", ["Relevance (if available)", "Most recent"], index=1)

    # ---------- Apply filters ----------
    df = base_df.copy()

    # date range on published_at
    df = df[
        (df["published_at"].dt.date >= start_date) &
        (df["published_at"].dt.date <= end_date)
    ].copy()

    if sel_cats:
        df = df[df["category"].fillna("General").isin(sel_cats)]
    if sel_src:
        df = df[df["source"].fillna("Unknown").isin(sel_src)]
    if sel_sent:
        df = df[df["sentiment_label"].isin(sel_sent)]
    if q:
        blob = (df["title"].fillna("") + " " + df["summary"].fillna("")).str.lower()
        df = df[blob.str.contains(q, na=False)]

    if sort_mode == "Most recent":
        df = df.sort_values("published_at", ascending=False)
    else:
        # If relevance isn't stored in DB, fallback
        if "relevance" in df.columns:
            df = df.sort_values(["relevance", "published_at"], ascending=[False, False])
        else:
            df = df.sort_values("published_at", ascending=False)

    # ---------- Header ----------
    latest_ts = df["published_at"].max()
    latest_txt = str(latest_ts) if pd.notna(latest_ts) else "N/A"
    st.markdown(
        f"""
        <div class="ai-header">
          <div class="ai-title">🗞️ Smart News • AI Analytics</div>
          <div class="ai-subtitle">
            Signal-driven overview of headlines • <span class="badge">Latest: {latest_txt}</span>
            &nbsp; <span class="badge">DB: {DB_PATH}</span>
            &nbsp; <span class="badge">Unseen-only: {"ON" if unseen_only else "OFF"}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- KPIs ----------
    pos_rate = (df["sentiment_label"] == "Positive").mean() if len(df) else 0.0
    neg_rate = (df["sentiment_label"] == "Negative").mean() if len(df) else 0.0
    avg_sent = df["sentiment_compound"].mean() if len(df) else 0.0

    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi">
            <div class="kpi-label">Headlines</div>
            <div class="kpi-value">{len(df)}</div>
            <div class="kpi-hint">After filters</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Sources</div>
            <div class="kpi-value">{df["source"].nunique()}</div>
            <div class="kpi-hint">Distinct publishers</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Categories</div>
            <div class="kpi-value">{df["category"].nunique()}</div>
            <div class="kpi-hint">Topic buckets</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Sentiment</div>
            <div class="kpi-value">{avg_sent:.2f}</div>
            <div class="kpi-hint">Positive {safe_pct(pos_rate)} • Negative {safe_pct(neg_rate)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Tabs ----------
    tab_overview, tab_trends, tab_sent, tab_explorer = st.tabs(
        ["📌 Overview", "📈 Trends", "😊 Sentiment", "🧭 Explorer"]
    )

    with tab_overview:
        st.markdown('<div class="section-title">Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Where the attention is in the selected window.</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.bar_chart(df["category"].fillna("General").value_counts().head(12))
        with c2:
            st.bar_chart(df["source"].fillna("Unknown").value_counts().head(12))

        st.markdown("---")
        st.markdown('<div class="section-title">Top Headlines</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Quick scan of the highest-signal items.</div>', unsafe_allow_html=True)

        topn = df.head(20).copy()
        topn["open"] = topn["link"].apply(lambda u: f"[Open]({u})" if isinstance(u, str) and u.startswith("http") else "")
        show_cols = ["published_at", "category", "source", "sentiment_label", "sentiment_compound", "title", "open"]
        for col in show_cols:
            if col not in topn.columns:
                topn[col] = ""
        st.dataframe(topn[show_cols], use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Actions</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Mark headlines as seen to focus only on new items.</div>', unsafe_allow_html=True)

        # Mark-as-seen tool
        if "content_hash" in df.columns and len(df) > 0:
            sample = df.head(100).copy()
            label_map = dict(zip(sample["content_hash"], sample["title"].astype(str).tolist()))
            selected = st.multiselect(
                "Select headlines to mark as seen",
                options=sample["content_hash"].tolist(),
                format_func=lambda h: label_map.get(h, h),
            )
            if st.button("✅ Mark selected as seen"):
                store.mark_seen(selected, seen=True)
                st.success(f"Marked {len(selected)} as seen.")
                st.rerun()

    with tab_trends:
        st.markdown('<div class="section-title">Trending Keywords</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Today vs average of previous 6 days (unique dates). Delta = trending up.</div>',
            unsafe_allow_html=True,
        )

        today_d = date.today()
        trend = trending_today_vs_last7days(store, today=today_d, top_k=20)

        if trend.empty:
            st.info("Not enough data yet to compute trends. Run the refresher daily for a week.")
        else:
            st.dataframe(trend, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Top Keywords in Current Filter</div>', unsafe_allow_html=True)

        kw = df_keyword_counts(df, top_k=20)
        if kw.empty:
            st.info("No keywords to display (empty after filters).")
        else:
            st.bar_chart(kw.set_index("keyword")["count"])

    with tab_sent:
        st.markdown('<div class="section-title">Sentiment Overview</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">VADER sentiment on title + summary.</div>', unsafe_allow_html=True)

        s1, s2 = st.columns(2)
        with s1:
            dist = df["sentiment_label"].value_counts()
            st.bar_chart(dist)

        with s2:
            # Altair-safe histogram (convert intervals to strings)
            bins = pd.cut(df["sentiment_compound"], bins=[-1, -0.6, -0.2, 0.2, 0.6, 1], include_lowest=True)
            hist = bins.value_counts().sort_index().reset_index()
            hist.columns = ["range", "count"]
            hist["range"] = hist["range"].astype(str)
            st.bar_chart(hist.set_index("range")["count"])

        st.markdown("---")
        st.markdown('<div class="section-title">Extremes</div>', unsafe_allow_html=True)

        a, b = st.columns(2)
        with a:
            neg = df.sort_values("sentiment_compound", ascending=True).head(10).copy()
            neg["open"] = neg["link"].apply(lambda u: f"[Open]({u})" if isinstance(u, str) and u.startswith("http") else "")
            st.caption("Most Negative")
            st.dataframe(
                neg[["sentiment_compound", "sentiment_label", "source", "title", "open"]],
                use_container_width=True,
                hide_index=True,
            )
        with b:
            pos = df.sort_values("sentiment_compound", ascending=False).head(10).copy()
            pos["open"] = pos["link"].apply(lambda u: f"[Open]({u})" if isinstance(u, str) and u.startswith("http") else "")
            st.caption("Most Positive")
            st.dataframe(
                pos[["sentiment_compound", "sentiment_label", "source", "title", "open"]],
                use_container_width=True,
                hide_index=True,
            )

    with tab_explorer:
        st.markdown('<div class="section-title">Headline Explorer</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Filter, scan, and open the underlying articles.</div>', unsafe_allow_html=True)

        n = st.slider("Rows", min_value=10, max_value=300, value=80, step=10)
        view = df.head(n).copy()
        view["open"] = view["link"].apply(lambda u: f"[Open]({u})" if isinstance(u, str) and u.startswith("http") else "")

        cols = ["published_at", "category", "source", "sentiment_label", "sentiment_compound", "title", "open"]
        for col in cols:
            if col not in view.columns:
                view[col] = ""

        st.dataframe(view[cols], use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Quick Reader</div>', unsafe_allow_html=True)

        for _, r in view.head(25).iterrows():
            title = str(r.get("title", ""))
            source = str(r.get("source", ""))
            with st.expander(f"{title} — {source}"):
                st.write(f"**Category:** {r.get('category', '')}")
                st.write(f"**Sentiment:** {r.get('sentiment_label', '')} ({float(r.get('sentiment_compound', 0.0)):.2f})")
                st.write(f"**Published:** {r.get('published_at', '')}")
                link = str(r.get("link", ""))
                if link:
                    st.markdown(f"**Link:** {link}")
                summary = str(r.get("summary", ""))
                if summary:
                    st.write(summary)


if __name__ == "__main__":
    app()