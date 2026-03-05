from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

DEFAULT_DB_PATH = "data/news.db"


def ensure_parent_dir(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def compute_content_hash(title: str, link: str, source: str) -> str:
    raw = f"{(title or '').strip().lower()}|{(link or '').strip().lower()}|{(source or '').strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class NewsStore:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        ensure_parent_dir(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        # timeout helps if Streamlit + scheduler access DB close together
        con = sqlite3.connect(self.db_path, timeout=20)
        con.row_factory = sqlite3.Row
        return con

    @staticmethod
    def _table_columns(con: sqlite3.Connection, table: str) -> Dict[str, str]:
        cols: Dict[str, str] = {}
        for r in con.execute(f"PRAGMA table_info({table});").fetchall():
            cols[str(r["name"])] = str(r["type"])
        return cols

    def _migrate(self, con: sqlite3.Connection) -> None:
        """
        Adds missing columns if DB was created with an older schema.
        Safe to run every startup.
        """
        cols = self._table_columns(con, "headlines")
        if not cols:
            return

        def add_col(name: str, coltype: str) -> None:
            if name not in cols:
                con.execute(f"ALTER TABLE headlines ADD COLUMN {name} {coltype};")

        # These are the columns your pipeline/dashboard expects
        add_col("published_at", "TEXT")
        add_col("sentiment_compound", "REAL")
        add_col("sentiment_label", "TEXT")
        add_col("first_seen_at", "TEXT")
        add_col("last_seen_at", "TEXT")
        add_col("seen", "INTEGER NOT NULL DEFAULT 0")

        # indexes (idempotent)
        con.execute("CREATE INDEX IF NOT EXISTS idx_headlines_published ON headlines(published_at);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_headlines_source ON headlines(source);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_headlines_category ON headlines(category);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_headlines_seen ON headlines(seen);")

    def _init_db(self) -> None:
        with self._connect() as con:
            # Better concurrent-read performance (Streamlit) + safer writes
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")

            con.execute(
                """
                CREATE TABLE IF NOT EXISTS headlines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    summary TEXT,
                    link TEXT NOT NULL,
                    source TEXT NOT NULL,
                    category TEXT,
                    published_at TEXT,              -- ISO8601
                    sentiment_compound REAL,
                    sentiment_label TEXT,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    seen INTEGER NOT NULL DEFAULT 0
                );
                """
            )

            # If DB existed already with old schema, upgrade it
            self._migrate(con)

    def upsert_from_df(self, df: pd.DataFrame) -> Dict[str, int]:
        if df.empty:
            return {"inserted": 0, "updated": 0}

        for col in ["title", "link", "source"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        ts = now_iso()
        inserted = 0
        updated = 0

        with self._connect() as con:
            # Ensure migrations applied even if db file was swapped in
            self._migrate(con)

            for _, r in df.iterrows():
                title = str(r.get("title", "")).strip()
                link = str(r.get("link", "")).strip()
                source = str(r.get("source", "")).strip()
                if not title or not link or not source:
                    continue

                content_hash = compute_content_hash(title, link, source)
                summary = str(r.get("summary", "") or "")
                category = str(r.get("category", "") or "General")

                # Your processor uses "published" column (not published_at)
                published = r.get("published")
                published_at = None
                if pd.notna(published):
                    published_at = pd.to_datetime(published).to_pydatetime().replace(microsecond=0).isoformat()

                sentiment_compound = r.get("sentiment_compound")
                sentiment_label = r.get("sentiment_label")

                con.execute(
                    """
                    INSERT INTO headlines (
                        content_hash, title, summary, link, source, category, published_at,
                        sentiment_compound, sentiment_label, first_seen_at, last_seen_at, seen
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    ON CONFLICT(content_hash) DO UPDATE SET
                        last_seen_at = excluded.last_seen_at,
                        summary = CASE
                            WHEN headlines.summary IS NULL OR headlines.summary = '' THEN excluded.summary
                            ELSE headlines.summary
                        END,
                        category = CASE
                            WHEN headlines.category IS NULL OR headlines.category = '' THEN excluded.category
                            ELSE headlines.category
                        END,
                        published_at = COALESCE(headlines.published_at, excluded.published_at),
                        sentiment_compound = COALESCE(headlines.sentiment_compound, excluded.sentiment_compound),
                        sentiment_label = COALESCE(headlines.sentiment_label, excluded.sentiment_label)
                    ;
                    """,
                    (
                        content_hash,
                        title,
                        summary,
                        link,
                        source,
                        category,
                        published_at,
                        float(sentiment_compound) if pd.notna(sentiment_compound) else None,
                        str(sentiment_label) if sentiment_label else None,
                        ts,
                        ts,
                    ),
                )

            # Better counts:
            row = con.execute("SELECT COUNT(*) AS c FROM headlines WHERE first_seen_at = ?;", (ts,)).fetchone()
            inserted = int(row["c"])
            # approximate updated
            updated = max(int(len(df)) - inserted, 0)

        return {"inserted": inserted, "updated": updated}

    def mark_seen(self, content_hashes: List[str], seen: bool = True) -> None:
        if not content_hashes:
            return
        val = 1 if seen else 0
        with self._connect() as con:
            con.executemany(
                "UPDATE headlines SET seen = ? WHERE content_hash = ?;",
                [(val, h) for h in content_hashes],
            )

    def query_headlines(
        self,
        since_iso: Optional[str] = None,
        until_iso: Optional[str] = None,
        categories: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        sentiments: Optional[List[str]] = None,
        unseen_only: bool = False,
        limit: int = 500,
        order_by: str = "published_at DESC",
    ) -> pd.DataFrame:
        where = []
        params: List[Any] = []

        if since_iso:
            where.append("published_at >= ?")
            params.append(since_iso)
        if until_iso:
            where.append("published_at <= ?")
            params.append(until_iso)
        if categories:
            where.append(f"category IN ({','.join(['?'] * len(categories))})")
            params.extend(categories)
        if sources:
            where.append(f"source IN ({','.join(['?'] * len(sources))})")
            params.extend(sources)
        if sentiments:
            where.append(f"sentiment_label IN ({','.join(['?'] * len(sentiments))})")
            params.extend(sentiments)
        if unseen_only:
            where.append("seen = 0")

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        sql = f"""
        SELECT
            content_hash, title, summary, link, source, category,
            published_at, sentiment_compound, sentiment_label,
            first_seen_at, last_seen_at, seen
        FROM headlines
        {where_sql}
        ORDER BY {order_by}
        LIMIT ?;
        """
        params.append(limit)

        with self._connect() as con:
            rows = con.execute(sql, params).fetchall()

        df = pd.DataFrame([dict(r) for r in rows])
        if not df.empty and "published_at" in df.columns:
            df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        return df