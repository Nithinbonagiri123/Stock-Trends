#!/usr/bin/env python3
"""
Train global LSTM–Transformer on Nifty-100 subset, then fine-tune on full downloaded universe.
For walk-forward validation helpers see training.walk_forward.
"""

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

from config import NIFTY100_PATH, PARQUET_PATH
from data.store import load_panel
from training.pipeline import finetune_on_universe, train_global_nifty100

logger = logging.getLogger(__name__)


def _load_nifty_symbols(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=PARQUET_PATH, help="Panel Parquet or HDF5 path")
    ap.add_argument("--nifty-list", type=Path, default=NIFTY100_PATH, help="One symbol per line")
    ap.add_argument("--global-only", action="store_true", help="Skip fine-tune stage")
    args = ap.parse_args()

    panel = load_panel(args.parquet)
    nifty = _load_nifty_symbols(args.nifty_list)
    train_global_nifty100(panel, nifty)
    if not args.global_only:
        finetune_on_universe(panel)
    logger.info("Training complete. Weights under artifacts/")


if __name__ == "__main__":
    main()
