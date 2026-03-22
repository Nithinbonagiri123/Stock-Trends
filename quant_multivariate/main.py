#!/usr/bin/env python3
"""
Multivariate LSTM + attention for BTC-USD (target) with ETH-USD and ALI=F as cross-asset inputs.

Usage (from quant_multivariate/):
  python3 main.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import SYMBOLS, YEARS
from data_loader import download_aligned
from train import load_and_train


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("quant_multivariate")

    logger.info("Downloading %s (%s years)…", SYMBOLS, YEARS)
    panel = download_aligned(SYMBOLS, years=YEARS)

    logger.info("Building features + walk-forward training (this may take a long time)…")
    result = load_and_train(panel)

    print("\n=== Walk-forward summary ===")
    print(json.dumps(result, indent=2, default=str))
    print(
        "\nBest checkpoint (highest val Sharpe seen during training):",
        result.get("model_path"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
