from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

from dateutil.tz import gettz

from .fetcher import load_config, NewsFetcher
from .processor import NewsProcessor, ProcessingOptions
from .summarizer import Summarizer, DigestOptions
from .scheduler import run_daily, store_run


def setup_logging(log_path: str) -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def build_and_save_digest(
    config_path: str = "config/sources.yaml",
    outputs_dir: str = "outputs",
    runs_dir: str = "data/runs",
    keywords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    tz_name = cfg.get("timezone", "Asia/Colombo")

    fetcher = NewsFetcher(timeout_s=12, max_retries=2)
    raw_items = fetcher.fetch_all(cfg)

    processor = NewsProcessor()
    df = processor.process(
        raw_items,
        ProcessingOptions(
            timezone=tz_name,
            today_only=True,
            keywords=keywords,
            near_dup_threshold=0.90,
            remove_clickbait=True,
        ),
    )

    # Save latest headlines and run snapshot
    Path("data").mkdir(parents=True, exist_ok=True)
    Path(runs_dir).mkdir(parents=True, exist_ok=True)

    # Latest file (dashboard reads this)
    df.to_csv("data/latest_headlines.csv", index=False, encoding="utf-8")
    df.to_json("data/latest_headlines.json", orient="records", force_ascii=False)

    # Per-run snapshot for trending keywords (last 7 runs)
    tz = gettz(tz_name)
    stamp = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
    run_headlines_path = str(Path(runs_dir) / f"headlines_{stamp}.csv")
    df.to_csv(run_headlines_path, index=False, encoding="utf-8")

    summarizer = Summarizer()
    digest = summarizer.build_digest(df, DigestOptions(timezone=tz_name, top_n=10))
    out_paths = summarizer.save_outputs(digest, outputs_dir, tz_name)

    run_record = {
        "timezone": tz_name,
        "keywords": keywords or [],
        "fetched_items": len(raw_items),
        "kept_items": int(len(df)),
        "outputs": out_paths,
        "headlines_snapshot": run_headlines_path,
    }

    run_record_path = store_run(runs_dir, run_record, tz_name)

    return {"run_record_path": run_record_path, **run_record}


def main():
    setup_logging("logs/app.log")

    import argparse

    parser = argparse.ArgumentParser(description="Smart News Headline Aggregator")
    parser.add_argument("--config", default="config/sources.yaml", help="Path to sources config YAML")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--schedule", action="store_true", help="Run daily at 07:00 in configured timezone")
    parser.add_argument("--keywords", nargs="*", default=None, help="Keywords to filter (e.g., AI economy sports)")
    args = parser.parse_args()

    def job():
        return build_and_save_digest(
            config_path=args.config,
            keywords=args.keywords,
        )

    # Run once
    if args.once:
        result = job()
        print("Digest generated:", result["outputs"])
        print("Run record saved:", result["run_record_path"])
        return

    # Scheduled run
    if args.schedule:
        cfg = load_config(args.config)
        tz_name = cfg.get("timezone", "Asia/Colombo")
        run_daily(job_fn=job, tz_name=tz_name, hour=7, minute=0)
        return

    # Default: run once
    result = job()
    print("Digest generated:", result["outputs"])
    print("Run record saved:", result["run_record_path"])


if __name__ == "__main__":
    main()