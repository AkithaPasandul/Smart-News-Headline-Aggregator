# Smart News Headline Aggregator

Fetches headlines from multiple RSS + optional JSON APIs, deduplicates, filters, categorizes, removes clickbait, and generates a daily digest (TXT/MD/HTML). Includes daily scheduling at 07:00.

## Requirements
- Python 3.10+

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
````

## Run once (today-only)

```bash
python -m news.main --once
```

With keyword filtering:

```bash
python -m news.main --once --keywords AI economy
```

## Run daily at 07:00 (Asia/Colombo)

```bash
python -m news.main --schedule
```

## Outputs

* `outputs/digest_YYYY-MM-DD.txt`
* `outputs/digest_YYYY-MM-DD.md`
* `outputs/digest_YYYY-MM-DD.html`

## Run history

Each run is stored in:

* `data/runs/run_YYYYMMDD_HHMMSS.json`

## Optional: enable NewsAPI

1. Create an API key at NewsAPI.org
2. Set environment variable:

   * Windows (PowerShell): `$env:NEWSAPI_KEY="YOUR_KEY"`
   * macOS/Linux: `export NEWSAPI_KEY="YOUR_KEY"`
3. In `config/sources.yaml`, set `enabled: true` for the NewsAPI source.

```

---

## 5) Explanation of each module (quick)

- **`fetcher.py`**: loads config, fetches RSS and JSON APIs with retries/timeouts, outputs `RawItem` list.
- **`processor.py`**: converts to DataFrame, normalizes dates, dedupes (exact + near), filters (today + keywords), removes clickbait, categorizes, relevance-scores, sorts.
- **`summarizer.py`**: builds digest grouped by category; exports TXT/MD/HTML.
- **`scheduler.py`**: APScheduler cron trigger; stores run metadata.
- **`main.py`**: CLI entrypoint; run-once or scheduled mode; logging.

---

## 6) Scalability & extendability notes (what to upgrade next)

If you plan to scale beyond a few hundred headlines/day:

1) **Faster near-duplicate detection**  
   Replace `SequenceMatcher` with MinHash/LSH (datasketch) or sentence embeddings + ANN.

2) **Persistent storage**  
   Add SQLite (or Postgres) to store every headline + hash, enabling “seen before” suppression across days.

3) **Trending keywords**  
   Daily TF-IDF over titles; keep top movers vs. last 7 days.

4) **Sentiment**  
   VADER (fast) or transformer sentiment (better, heavier) applied per headline/summary.

5) **Dashboard** (Streamlit)  
   - Filters: date, category, keyword  
   - Charts: trending terms, source distribution  
   - Table: clickable links

---