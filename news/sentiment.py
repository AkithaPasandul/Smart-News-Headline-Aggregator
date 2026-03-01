from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


_ANALYZER = SentimentIntensityAnalyzer()


def add_vader_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - sentiment_compound: float [-1,1]
      - sentiment_label: Negative/Neutral/Positive
    """
    if df.empty:
        df["sentiment_compound"] = []
        df["sentiment_label"] = []
        return df

    df = df.copy()
    text = (df["title"].fillna("") + ". " + df["summary"].fillna("")).astype(str)

    df["sentiment_compound"] = text.apply(lambda s: _ANALYZER.polarity_scores(s)["compound"])

    def label(x: float) -> str:
        if x >= 0.05:
            return "Positive"
        if x <= -0.05:
            return "Negative"
        return "Neutral"

    df["sentiment_label"] = df["sentiment_compound"].apply(label)
    return df