"""
APScheduler: weekdays at configured US time (default ~1h before equity close).
"""

from __future__ import annotations

import logging
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from .config_auto import SCHEDULE_HOUR, SCHEDULE_MINUTE, SCHEDULE_TIMEZONE
from .pipeline import run_pipeline_with_retry

logger = logging.getLogger(__name__)


def run_scheduler() -> None:
    tz = ZoneInfo(SCHEDULE_TIMEZONE)
    sched = BlockingScheduler(timezone=tz)
    sched.add_job(
        run_pipeline_with_retry,
        CronTrigger(
            day_of_week="mon-fri",
            hour=SCHEDULE_HOUR,
            minute=SCHEDULE_MINUTE,
            timezone=tz,
        ),
        id="quant_daily_job",
        replace_existing=True,
    )
    logger.info(
        "Scheduler running Mon–Fri at %02d:%02d %s (Ctrl+C to stop)",
        SCHEDULE_HOUR,
        SCHEDULE_MINUTE,
        SCHEDULE_TIMEZONE,
    )
    sched.start()
