from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, date
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from dateutil import parser as date_parser
from dateutil.tz import gettz

from .fetcher import RawItem


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessingOptions:
    timezone: str = "Asia/Colombo"
    today_only: bool = True
    keywords: Optional[List[str]] = None  # e.g. ["AI", "economy"]
    max_items_per_source: int = 200
    near_dup_threshold: float = 0.90
    remove_clickbait: bool = True


CATEGORY_RULES: Dict[str, List[str]] = {
    "Technology": ["ai", "artificial intelligence", "machine learning", "cyber", "security", "software", "chip", "robot"],
    "Business": ["economy", "market", "stocks", "inflation", "bank", "trade", "oil", "finance", "startup"],
    "Sports": ["match", "tournament", "league", "goal", "cricket", "football", "soccer", "tennis", "nba", "f1"],
    "Politics": ["election", "parliament", "president", "minister", "policy", "government", "senate", "congress"],
    "World": ["war", "conflict", "united nations", "refugee", "sanction", "border", "summit", "diplomacy"],
    "Health": ["health", "disease", "virus", "hospital", "vaccine", "medicine", "who"],
}


CLICKBAIT_PATTERNS = [
    r"\byou won['’]t believe\b",
    r"\bwhat happens next\b",
    r"\b(shocking|unbelievable|mind[- ]blowing)\b",
    r"!!!+",
    r"\bthis is why\b",
    r"\btop\s+\d+\b",
]


def to_dataframe(items: List[RawItem], tz_name: str) -> pd.DataFrame:
    tz = gettz(tz_name)

    rows = []
    for it in items:
        rows.append(
            {
                "title": it.title,
                "source": it.source,
                "published_raw": it.published,
                "published": _parse_date(it.published, tz),
                "link": it.link,
                "summary": it.summary,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Drop items with no parsed date (keep, but push to oldest)
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=False)
    df["published"] = df["published"].fillna(pd.Timestamp("1970-01-01", tz=tz))

    return df


def _parse_date(published: Optional[str], tz) -> Optional[datetime]:
    if not published:
        return None
    try:
        dt = date_parser.parse(published)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        return dt.astimezone(tz)
    except Exception:
        return None


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)  # remove punctuation
    return s


def is_clickbait(title: str) -> bool:
    t = title.strip()
    if not t:
        return False
    # heuristic 1: too many caps
    letters = [c for c in t if c.isalpha()]
    if letters:
        cap_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if cap_ratio > 0.65 and len(letters) > 12:
            return True
    # heuristic 2: patterns
    low = t.lower()
    for pat in CLICKBAIT_PATTERNS:
        if re.search(pat, low):
            return True
    # heuristic 3: too much punctuation
    if t.count("!") >= 2 or t.count("?") >= 2:
        return True
    return False


def categorize(title: str, summary: str) -> str:
    text = f"{title} {summary}".lower()
    for cat, keys in CATEGORY_RULES.items():
        if any(k in text for k in keys):
            return cat
    return "General"


def keyword_filter_mask(df: pd.DataFrame, keywords: Optional[List[str]]) -> pd.Series:
    if not keywords:
        return pd.Series([True] * len(df), index=df.index)

    kws = [k.lower().strip() for k in keywords if k.strip()]
    if not kws:
        return pd.Series([True] * len(df), index=df.index)

    combined = (df["title"].fillna("") + " " + df["summary"].fillna("")).str.lower()
    return combined.apply(lambda s: any(k in s for k in kws))


def today_only_mask(df: pd.DataFrame, tz_name: str) -> pd.Series:
    tz = gettz(tz_name)
    today = datetime.now(tz).date()
    published_dates = df["published"].dt.tz_convert(tz).dt.date
    return published_dates == today


def dedupe_exact(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title_norm"] = df["title"].fillna("").map(normalize_text)
    before = len(df)
    df = df.drop_duplicates(subset=["title_norm"], keep="first")
    logger.info("Exact dedupe: %d -> %d", before, len(df))
    return df


def dedupe_near(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Simple O(n^2) near-duplicate removal (fine for a few hundred headlines).
    For thousands+, replace with MinHash/LSH or embeddings + ANN.
    """
    if df.empty:
        return df

    df = df.copy()
    titles = df["title"].fillna("").tolist()
    norms = [normalize_text(t) for t in titles]

    keep = [True] * len(df)
    for i in range(len(df)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(df)):
            if not keep[j]:
                continue
            sim = SequenceMatcher(None, norms[i], norms[j]).ratio()
            if sim >= threshold:
                keep[j] = False

    before = len(df)
    df = df.loc[keep].reset_index(drop=True)
    logger.info("Near dedupe(th=%.2f): %d -> %d", threshold, before, len(df))
    return df


def score_relevance(df: pd.DataFrame, keywords: Optional[List[str]]) -> pd.DataFrame:
    """
    Lightweight scoring: keyword hits + recency.
    """
    df = df.copy()
    text = (df["title"].fillna("") + " " + df["summary"].fillna("")).str.lower()

    kw_score = 0
    if keywords:
        kws = [k.lower().strip() for k in keywords if k.strip()]
        kw_score = text.apply(lambda s: sum(1 for k in kws if k in s))

    # recency: newer => bigger
    max_ts = df["published"].max()
    age_minutes = (max_ts - df["published"]).dt.total_seconds() / 60.0
    recency_score = (-age_minutes).fillna(0)

    df["relevance"] = kw_score + (recency_score / 1440.0)  # scale minutes -> days
    return df


class NewsProcessor:
    def process(self, items: List[RawItem], opts: ProcessingOptions) -> pd.DataFrame:
        df = to_dataframe(items, opts.timezone)
        if df.empty:
            return df

        # cap per source (avoid one noisy feed dominating)
        df = (
            df.sort_values("published", ascending=False)
            .groupby("source", as_index=False)
            .head(opts.max_items_per_source)
            .reset_index(drop=True)
        )

        df = dedupe_exact(df)
        df = dedupe_near(df, threshold=opts.near_dup_threshold)

        if opts.today_only:
            df = df[today_only_mask(df, opts.timezone)].copy()

        # clickbait filter
        if opts.remove_clickbait:
            mask = ~df["title"].fillna("").apply(is_clickbait)
            df = df[mask].copy()

        # keyword filter
        df = df[keyword_filter_mask(df, opts.keywords)].copy()

        # categorize
        df["category"] = df.apply(lambda r: categorize(r["title"], r["summary"]), axis=1)

        # relevance + sort
        df = score_relevance(df, opts.keywords)
        df = df.sort_values(["relevance", "published"], ascending=[False, False]).reset_index(drop=True)

        return df