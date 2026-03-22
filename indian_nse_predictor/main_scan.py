#!/usr/bin/env python3
"""Run one market scan (rank + optional Telegram chart). Use run_scheduler.py for 3:45 PM IST."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from config import PARQUET_PATH
from scanner.daily_scanner import run_daily_scan


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=PARQUET_PATH, help="Panel path (Parquet or HDF5)")
    ap.add_argument("--no-telegram", action="store_true", help="Print results only")
    args = ap.parse_args()

    top, risk = run_daily_scan(panel_path=args.parquet, send_telegram=not args.no_telegram)
    print(top.to_string())
    print(risk)


if __name__ == "__main__":
    main()
