#!/usr/bin/env python3
"""
Download NSE equity history (yfinance .NS), optional bhavcopy delivery for latest session,
and persist to Parquet (and optionally compressed HDF5).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    DOWNLOAD_WORKERS,
    HDF5_PATH,
    MAX_SYMBOLS,
    NIFTY100_PATH,
    PARQUET_PATH,
    SAVE_HDF5,
)
from data.bhavcopy import fetch_bhavcopy_delivery, merge_delivery_into_daily
from data.downloader import download_parallel
from data.store import append_to_hdf5, frames_to_parquet
from data.symbols import load_or_fetch_symbols

logger = logging.getLogger(__name__)


def _nifty_symbols_ns() -> list[str]:
    """Nifty list as Yahoo tickers (must overlap global training universe)."""
    if not NIFTY100_PATH.exists():
        return []
    lines = NIFTY100_PATH.read_text(encoding="utf-8").splitlines()
    return [f"{ln.strip().upper()}.NS" for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def build_download_symbol_list(max_symbols: int) -> list[str]:
    """Prefer Nifty names first, then fill from full NSE list up to max_symbols."""
    nifty = _nifty_symbols_ns()
    nifty_set = set(nifty)
    rest = [s for s in load_or_fetch_symbols() if s not in nifty_set]
    merged: list[str] = []
    seen: set[str] = set()
    for s in nifty + rest:
        if max_symbols > 0 and len(merged) >= max_symbols:
            break
        if s not in seen:
            seen.add(s)
            merged.append(s)
    return merged


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Download NSE daily history to Parquet.")
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing Parquet/HDF5 store before writing (use after a bad merge).",
    )
    args = ap.parse_args()
    if args.fresh:
        for p in (PARQUET_PATH, HDF5_PATH):
            if p.exists():
                p.unlink()
                logger.info("Removed %s", p)

    syms = build_download_symbol_list(MAX_SYMBOLS)
    logger.info("Downloading %s symbols (MAX_SYMBOLS=%s)", len(syms), MAX_SYMBOLS)
    frames = download_parallel(syms, workers=DOWNLOAD_WORKERS)
    bhav = fetch_bhavcopy_delivery()
    merged = [merge_delivery_into_daily(df, bhav) for df in frames]
    frames_to_parquet(merged, PARQUET_PATH)
    if SAVE_HDF5:
        append_to_hdf5(merged, HDF5_PATH)
    logger.info("Done. Parquet → %s", PARQUET_PATH)


if __name__ == "__main__":
    main()
