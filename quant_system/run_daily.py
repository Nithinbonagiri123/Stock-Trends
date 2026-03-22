#!/usr/bin/env python3
"""
Automated daily quant pipeline.

Usage (from the `quant_system` directory):
  pip install -r requirements.txt
  python run_daily.py --once          # single run now
  python run_daily.py                 # APScheduler loop (weekdays)

Environment: copy `.env.example` to `.env` and set secrets (Telegram / email optional).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure quant_system is on path when executed as a script
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None  # type: ignore

    if load_dotenv:
        load_dotenv(_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Stock prediction automation")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run the pipeline once and exit (no scheduler)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        if args.once:
            from automation.pipeline import run_pipeline_with_retry

            run_pipeline_with_retry()
        else:
            from automation.scheduler import run_scheduler

            run_scheduler()
    except Exception:
        logging.exception("Pipeline or scheduler failed.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
