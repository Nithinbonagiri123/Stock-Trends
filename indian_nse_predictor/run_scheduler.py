#!/usr/bin/env python3
"""
Blocking scheduler: run market scan daily at SCAN_HOUR:SCAN_MINUTE IST (default 15:45).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from config import SCAN_HOUR, SCAN_MINUTE, TZ_IST
from scanner.daily_scanner import run_daily_scan

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    def job() -> None:
        try:
            run_daily_scan(send_telegram=True)
        except Exception:
            logger.exception("Daily scan failed")

    sched = BlockingScheduler(timezone=TZ_IST)
    sched.add_job(
        job,
        "cron",
        hour=SCAN_HOUR,
        minute=SCAN_MINUTE,
        timezone=TZ_IST,
        id="nse_scan",
        replace_existing=True,
    )
    logger.info("Scheduler started: %s:%02d %s (Ctrl+C to stop)", SCAN_HOUR, SCAN_MINUTE, TZ_IST)
    sched.start()


if __name__ == "__main__":
    main()
