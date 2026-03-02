from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st

from news.sentiment import add_vader_sentiment
from news.main import run_once_from_dashboard

DATA_LATEST = Path("data/latest_headlines.csv")
RUNS_DIR = Path("data/runs")

# Small stopword list (extend anytime)
STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "as", "at", "by",
    "from", "is", "are", "was", "were", "be", "been", "it", "that", "this", "these", "those",
    "after", "before", "over", "under", "into", "about", "amid", "says", "say", "new", "latest",
}


def load_latest() -> pd.DataFrame:
    if not DATA_LATEST.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_LATEST)

    if "published" in df.columns:
        df["published"] = pd.to_datetime(df["published"], errors="coerce")

    # Ensure core columns exist
    for col in ["title", "source", "summary", "link", "category", "relevance"]:
        if col not in df.columns:
            df[col] = ""

    # Add sentiment if missing
    if "sentiment_compound" not in df.columns or "sentiment_label" not in df.columns:
        df = add_vader_sentiment(df)

    return df


def list_last_run_snapshots(limit: int = 7) -> List[Path]:
    if not RUNS_DIR.exists():
        return []
    files = sorted(RUNS_DIR.glob("headlines_*.csv"), reverse=True)
    return files[:limit]


def tokenize(text: str) -> List[str]:
    text = text.lower()
    words = re.findall(r"[a-z0-9]+", text)
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]


def keyword_counts(df: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["keyword", "count"])

    blob = (df["title"].fillna("") + " " + df["summary"].fillna("")).astype(str)
    c = Counter()
    for s in blob.tolist():
        c.update(tokenize(s))

    common = c.most_common(top_k)
    return pd.DataFrame(common, columns=["keyword", "count"])


def trending_today_vs_last7(today_df: pd.DataFrame, snapshot_paths: List[Path], top_k: int = 15) -> pd.DataFrame:
    """
    Compare today's keyword counts vs average of previous snapshots (excluding today if it is included).
    Returns: keyword, today_count, prev_avg, delta
    """
    today_counts = keyword_counts(today_df, top_k=200).set_index("keyword")["count"].to_dict()

    prev_counts_total = Counter()
    prev_days = 0

    # Use previous 6 snapshots (if we have 7, treat current latest as "today" and compare against the rest)
    # We'll just use all snapshots and compute an average; it’s robust even if today is included (still useful).
    for p in snapshot_paths:
        try:
            df = pd.read_csv(p)
            for col in ["title", "summary"]:
                if col not in df.columns:
                    df[col] = ""
            counts = keyword_counts(df, top_k=300)
            prev_counts_total.update(dict(zip(counts["keyword"], counts["count"])))
            prev_days += 1
        except Exception:
            continue

    if prev_days == 0:
        prev_days = 1

    # Build trend table for union of top today + top prev
    candidate_keywords = set(list(today_counts.keys()))
    # also include top prev
    for kw, _ in prev_counts_total.most_common(50):
        candidate_keywords.add(kw)

    rows = []
    for kw in candidate_keywords:
        today = float(today_counts.get(kw, 0))
        prev_avg = float(prev_counts_total.get(kw, 0)) / prev_days
        rows.append((kw, today, prev_avg, today - prev_avg))

    trend = pd.DataFrame(rows, columns=["keyword", "today_count", "prev_avg", "delta"])
    trend = trend.sort_values(["delta", "today_count"], ascending=[False, False]).head(top_k).reset_index(drop=True)
    return trend


def app():
    st.set_page_config(page_title="Smart News Dashboard", layout="wide")
    st.title("🗞️ Smart News Headline Aggregator Dashboard")
    
    # ---- Refresh controls ----
    st.sidebar.header("Actions")
    kw_input = st.sidebar.text_input("Refresh keywords (optional)", value="").strip()
    kw_list = [k for k in kw_input.split() if k] if kw_input else None
    
    if st.sidebar.button("🔄 Refresh News Now"):
        with st.spinner("Fetching & processing news..."):
            try:
                result = run_once_from_dashboard(keywords=kw_list)
                st.sidebar.success("Updated! Reloading dashboard...")
                st.session_state["last_refresh_result"] = result
                st.rerun()  # reload UI with new data
            except Exception as e:
                st.sidebar.error(f"Refresh failed: {e}")

    # ---- Live refresh controls ----
    st.sidebar.header("Live Refresh")
    refresh_min = st.sidebar.number_input("Auto-reload every (minutes)", min_value=0, max_value=120, value=5, step=1)
    if refresh_min > 0:
        # Streamlit built-in auto refresh
        st_autorefresh = getattr(st, "autorefresh", None)
        if callable(st_autorefresh):
            st.autorefresh(interval=int(refresh_min * 60 * 1000), key="auto_refresh")
        else:
            # fallback for older streamlit
            st.sidebar.info("Upgrade Streamlit to use autorefresh (st.autorefresh).")

    st.caption(f"Last UI refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df = load_latest()
    if df.empty:
        st.warning("No data found. Run:\n\n`python -m news.main --once`\n\nThen refresh.")
        return

    # ---- Sidebar filters ----
    st.sidebar.header("Filters")

    # Category
    cats = sorted(df["category"].fillna("General").unique().tolist())
    sel_cats = st.sidebar.multiselect("Category", cats, default=cats)

    # Source
    sources = sorted(df["source"].fillna("Unknown").unique().tolist())
    sel_src = st.sidebar.multiselect("Source", sources, default=sources)

    # Sentiment
    sentiments = ["Positive", "Neutral", "Negative"]
    sel_sent = st.sidebar.multiselect("Sentiment", sentiments, default=sentiments)

    # Keyword search
    q = st.sidebar.text_input("Search keyword", value="").strip().lower()

    filtered = df.copy()
    if sel_cats:
        filtered = filtered[filtered["category"].isin(sel_cats)]
    if sel_src:
        filtered = filtered[filtered["source"].isin(sel_src)]
    if sel_sent:
        filtered = filtered[filtered["sentiment_label"].isin(sel_sent)]
    if q:
        blob = (filtered["title"].fillna("") + " " + filtered["summary"].fillna("")).str.lower()
        filtered = filtered[blob.str.contains(q, na=False)]

    # Sort
    sort_mode = st.sidebar.selectbox("Sort by", ["Relevance (desc)", "Most recent"], index=0)
    if sort_mode == "Most recent":
        filtered = filtered.sort_values("published", ascending=False)
    else:
        if "relevance" in filtered.columns:
            filtered = filtered.sort_values(["relevance", "published"], ascending=[False, False])
        else:
            filtered = filtered.sort_values("published", ascending=False)

    # ---- KPIs ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Headlines", len(filtered))
    c2.metric("Sources", filtered["source"].nunique())
    c3.metric("Categories", filtered["category"].nunique())
    c4.metric("Latest", str(filtered["published"].max()) if "published" in filtered.columns else "-")

    st.divider()

    # ---- Charts row 1 ----
    left, right = st.columns(2)
    with left:
        st.subheader("Headlines by Category")
        st.bar_chart(filtered["category"].fillna("General").value_counts().head(12))

    with right:
        st.subheader("Headlines by Source")
        st.bar_chart(filtered["source"].fillna("Unknown").value_counts().head(12))

    st.divider()

    # ---- Sentiment charts ----
    s1, s2 = st.columns(2)
    with s1:
        st.subheader("Sentiment Distribution")
        st.bar_chart(filtered["sentiment_label"].value_counts())

    with s2:
        st.subheader("Sentiment (Compound) Histogram")
        # simple histogram via value counts of binned ranges
        bins = pd.cut(
            filtered["sentiment_compound"],
            bins=[-1, -0.6, -0.2, 0.2, 0.6, 1],
            include_lowest=True,
            )
        
        hist = bins.value_counts().sort_index()
        hist_df = hist.reset_index()
        hist_df.columns = ["range", "count"]
        
        # Convert Interval objects to strings (Altair-safe)
        hist_df["range"] = hist_df["range"].astype(str)
        
        st.bar_chart(hist_df.set_index("range")["count"])

    st.divider()

    # ---- Trending keywords (today vs last 7 runs) ----
    st.subheader("📈 Trending Keywords (Today vs Last 7 Runs)")
    snapshots = list_last_run_snapshots(limit=7)
    trend = trending_today_vs_last7(df, snapshots, top_k=15)
    st.dataframe(trend, use_container_width=True, hide_index=True)

    st.caption("Delta = today_count − avg(previous runs). Increase means it’s trending up.")

    st.divider()

    # ---- Table ----
    st.subheader("Headlines")
    show_n = st.slider("Rows to display", 10, 200, 50, 10)
    view = filtered.head(show_n).copy()
    view["open"] = view["link"].apply(lambda u: f"[Open]({u})" if isinstance(u, str) and u.startswith("http") else "")

    cols = ["published", "category", "source", "sentiment_label", "sentiment_compound", "title", "open"]
    for c in cols:
        if c not in view.columns:
            view[c] = ""

    st.dataframe(view[cols], use_container_width=True, hide_index=True)

    st.divider()

    # ---- Quick Reader ----
    st.subheader("Quick Reader")
    for _, r in view.iterrows():
        title = str(r.get("title", ""))
        source = str(r.get("source", ""))
        with st.expander(f"{title} — {source}"):
            st.write(f"**Category:** {r.get('category', '')}")
            st.write(f"**Sentiment:** {r.get('sentiment_label', '')} ({r.get('sentiment_compound', '')})")
            st.write(f"**Published:** {r.get('published', '')}")
            link = str(r.get("link", ""))
            if link:
                st.markdown(f"**Link:** {link}")
            summary = str(r.get("summary", ""))
            if summary:
                st.write(summary)


if __name__ == "__main__":
    app()