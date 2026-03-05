# 🗞️ Smart News Headline Aggregator

A production-ready Python application that fetches headlines from multiple news sources (RSS + APIs), cleans and deduplicates them, filters intelligently, performs sentiment analysis, detects trending keywords and generates a daily digest with an interactive dashboard.

---

## 🚀 Features

### 📰 Data Collection
- Fetches headlines from 5+ RSS sources
- Supports JSON News APIs (optional)
- Extracts:
  - Title
  - Source
  - Published date
  - Summary
  - Link
- Network retry + timeout handling

---

### 🧠 Intelligent Processing
- Converts to structured pandas DataFrame
- Date normalization
- Exact + near-duplicate removal
- Keyword filtering
- Clickbait detection (rule-based)
- Topic categorization
- Relevance scoring

---

### 😊 Sentiment Analysis
- VADER sentiment scoring
- Compound score (-1 to +1)
- Labels:
  - Positive
  - Neutral
  - Negative
- Sentiment charts in dashboard

---

### 📈 Trending Keyword Detection
- Compares today's keywords vs last 7 runs
- Calculates:
  - today_count
  - previous average
  - delta (trend direction)

---

### 📅 Daily Digest Output
Generated in:
- TXT
- Markdown
- Email-ready HTML

Example:

```

📅 Daily News Digest – 2026-03-01

🔹 Technology

* AI breakthrough announced (BBC)

🔹 Business

* Global markets rally (Reuters)

```

---

### 📊 Interactive Streamlit Dashboard
- Live auto-refresh (configurable minutes)
- Filter by:
  - Category
  - Source
  - Sentiment
  - Keyword
- Sentiment distribution chart
- Trending keyword table
- Headline explorer with quick reader
- Auto-loads latest processed data

---

### ⏰ Automation
- Scheduled daily run at 07:00 (configurable)
- Logs stored locally
- Per-run snapshot stored
- Run metadata saved as JSON

---

## 📂 Project Structure

```

smart-news-headline-aggregator/
│
├── news/
│   ├── **init**.py
│   ├── fetcher.py
│   ├── processor.py
│   ├── summarizer.py
│   ├── scheduler.py
│   ├── sentiment.py
│   ├── dashboard.py
│   └── main.py
│
├── config/
│   └── sources.yaml
│
├── data/
│   ├── latest_headlines.csv
│   └── runs/
│
├── outputs/
├── logs/
├── pyproject.toml
├── requirements.txt
└── README.md

````

---

## 🛠 Installation

### 1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd smart-news-headline-aggregator
````

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

If using requirements.txt:

```bash
pip install -r requirements.txt
```

If using pyproject.toml:

```bash
pip install -e ".[dashboard]"
```

---

## ▶ Run the Aggregator (Once)

```bash
python -m news.main --once
```

Optional keyword filter:

```bash
python -m news.main --once --keywords AI economy sports
```

Outputs saved to:

```
outputs/
data/latest_headlines.csv
data/runs/
```

---

## ⏰ Run Daily Scheduled Job

```bash
python -m news.main --schedule
```

Runs every day at 07:00 (configured timezone).

---

## 📊 Run the Dashboard

```bash
streamlit run news/dashboard.py
```

Dashboard Features:

* Live auto-refresh (sidebar setting)
* Sentiment charts
* Trending keywords (today vs last 7 runs)
* Headline explorer
* Quick reader mode

---

## 🔐 Optional: Enable NewsAPI

1. Create API key at [https://newsapi.org](https://newsapi.org)
2. Set environment variable:

**Windows (PowerShell):**

```powershell
$env:NEWSAPI_KEY="YOUR_KEY"
```

**Mac/Linux:**

```bash
export NEWSAPI_KEY="YOUR_KEY"
```

3. Enable in `config/sources.yaml`

---

## 🧩 Architecture Overview

```
RSS/API
   ↓
Fetcher (retry + timeout)
   ↓
Processor
   - Normalize dates
   - Deduplicate
   - Categorize
   - Score relevance
   - Filter clickbait
   ↓
Sentiment (VADER)
   ↓
Summarizer (TXT/MD/HTML)
   ↓
Snapshots stored
   ↓
Streamlit Dashboard
```

---

## 📈 Scalability Roadmap

Planned improvements:

* SQLite or PostgreSQL storage
* MinHash/LSH near-duplicate detection
* Embedding-based similarity
* Transformer sentiment model
* Trend detection by unique dates (last 7 days instead of runs)
* Docker deployment
* GitHub Actions CI/CD
* Email automation

---

## 🧠 Design Principles

* Modular structure
* Separation of concerns
* Clean logging
* Extendable configuration
* Dashboard-friendly data persistence
* Production-aware error handling

---

## 📜 License

MIT License

```

