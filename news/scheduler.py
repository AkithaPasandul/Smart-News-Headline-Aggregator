from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Any

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dateutil.tz import gettz


logger = logging.getLogger(__name__)


def store_run(runs_dir: str, payload: Dict[str, Any], tz_name: str) -> str:
    Path(runs_dir).mkdir(parents=True, exist_ok=True)
    tz = gettz(tz_name)
    stamp = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
    path = Path(runs_dir) / f"run_{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def run_daily(job_fn: Callable[[], Dict[str, Any]], tz_name: str, hour: int = 7, minute: int = 0):
    scheduler = BlockingScheduler(timezone=tz_name)
    trigger = CronTrigger(hour=hour, minute=minute, timezone=tz_name)

    def _wrapped():
        logger.info("Scheduled job started")
        try:
            result = job_fn()
            logger.info("Scheduled job success")
            return result
        except Exception as e:
            logger.exception("Scheduled job failed: %s", e)
            raise

    scheduler.add_job(_wrapped, trigger, id="daily_digest", replace_existing=True)
    logger.info("Scheduler started. Daily run at %02d:%02d (%s)", hour, minute, tz_name)
    scheduler.start()