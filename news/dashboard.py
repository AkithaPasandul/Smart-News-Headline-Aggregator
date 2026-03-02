from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from news.sentiment import add_vader_sentiment
from news.main import run_once_from_dashboard


DATA_LATEST = Path("data/latest_headlines.csv")
RUNS_DIR = Path("data/runs")

STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","as","at","by",
    "from","is","are","was","were","be","been","it","that","this","these","those",
    "after","before","over","under","into","about","amid","says","say","new","latest"
}


# Data Loading 
def load_data():
    if not DATA_LATEST.exists():
        return pd.DataFrame()

    df = pd.read_csv(DATA_LATEST)

    if "published" in df.columns:
        df["published"] = pd.to_datetime(df["published"], errors="coerce")

    if "sentiment_label" not in df.columns:
        df = add_vader_sentiment(df)

    return df


# Keyword Extraction
def tokenize(text: str):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if len(w) > 3 and w not in STOPWORDS]


def keyword_counts(df):
    blob = (df["title"].fillna("") + " " + df["summary"].fillna("")).astype(str)
    counter = Counter()

    for s in blob:
        counter.update(tokenize(s))

    return pd.DataFrame(counter.most_common(15), columns=["keyword","count"])


# Dashboard App
def app():
    st.set_page_config(layout="wide")

    # Header
    st.title("🗞️ Smart News Intelligence Dashboard")
    st.caption("Real-time headline aggregation, sentiment analysis, and trend detection")

    col1, col2 = st.columns([6, 2])

    with col1:
        st.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    with col2:
        if st.button("🔄 Refresh News"):
            with st.spinner("Updating news..."):
                run_once_from_dashboard()
                st.success("News Updated")
                st.rerun()

    st.divider()

    # Sidebar Filters
    st.sidebar.header("Filters")

    df = load_data()

    if df.empty:
        st.warning("No data found. Please refresh news.")
        return

    categories = sorted(df["category"].dropna().unique())
    selected_categories = st.sidebar.multiselect("Category", categories, default=categories)

    sources = sorted(df["source"].dropna().unique())
    selected_sources = st.sidebar.multiselect("Source", sources, default=sources)

    sentiments = ["Positive", "Neutral", "Negative"]
    selected_sentiments = st.sidebar.multiselect("Sentiment", sentiments, default=sentiments)

    keyword_search = st.sidebar.text_input("Search keyword")

    auto_refresh = st.sidebar.number_input("Auto Refresh (minutes)", min_value=0, max_value=60, value=0)

    if auto_refresh > 0:
        st.autorefresh(interval=auto_refresh * 60000)

    # Apply filters
    filtered = df.copy()

    if selected_categories:
        filtered = filtered[filtered["category"].isin(selected_categories)]

    if selected_sources:
        filtered = filtered[filtered["source"].isin(selected_sources)]

    if selected_sentiments:
        filtered = filtered[filtered["sentiment_label"].isin(selected_sentiments)]

    if keyword_search:
        blob = (filtered["title"] + " " + filtered["summary"]).str.lower()
        filtered = filtered[blob.str.contains(keyword_search.lower())]

    # KPIs Section
    st.subheader("📊 Overview")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Headlines", len(filtered))
    k2.metric("Sources", filtered["source"].nunique())
    k3.metric("Categories", filtered["category"].nunique())
    k4.metric("Positive %", f"{(filtered['sentiment_label']=='Positive').mean()*100:.1f}%")

    st.divider()

    # Charts Section
    st.subheader("📈 Distribution Analysis")

    c1, c2 = st.columns(2)

    with c1:
        st.write("Headlines by Category")
        st.bar_chart(filtered["category"].value_counts())

    with c2:
        st.write("Headlines by Source")
        st.bar_chart(filtered["source"].value_counts().head(10))

    st.divider()

    # Sentiment Analysis
    st.subheader("😊 Sentiment Analysis")

    s1, s2 = st.columns(2)

    with s1:
        st.write("Sentiment Distribution")
        st.bar_chart(filtered["sentiment_label"].value_counts())

    with s2:
        st.write("Sentiment Score Spread")
        bins = pd.cut(filtered["sentiment_compound"], bins=5)
        hist = bins.value_counts().sort_index().reset_index()
        hist.columns = ["range","count"]
        hist["range"] = hist["range"].astype(str)
        st.bar_chart(hist.set_index("range")["count"])

    st.divider()

    # Trending Keywords
    st.subheader("🔥 Trending Keywords")

    trend_df = keyword_counts(filtered)
    st.bar_chart(trend_df.set_index("keyword"))

    st.divider()

    # Headlines Table
    st.subheader("📰 Headlines")

    display_cols = [
        "published",
        "category",
        "source",
        "sentiment_label",
        "title"
    ]

    st.dataframe(filtered[display_cols], use_container_width=True)

    st.divider()

    # Quick Reader
    st.subheader("📖 Quick Reader")

    for _, row in filtered.head(20).iterrows():
        with st.expander(f"{row['title']} — {row['source']}"):
            st.write(f"Category: {row['category']}")
            st.write(f"Sentiment: {row['sentiment_label']} ({row['sentiment_compound']:.2f})")
            st.write(row["summary"])
            st.markdown(f"[Open Article]({row['link']})")


if __name__ == "__main__":
    app()