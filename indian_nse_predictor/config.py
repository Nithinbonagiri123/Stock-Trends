"""Paths, market rules, and scheduling (IST)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_store"
PARQUET_PATH = DATA_DIR / "nse_daily.parquet"
HDF5_PATH = DATA_DIR / "nse_daily.h5"
SYMBOLS_CACHE = DATA_DIR / "symbols.json"
SAVE_HDF5 = os.environ.get("SAVE_HDF5", "").strip() in ("1", "true", "yes")
SECTOR_MAP_PATH = ROOT / "data" / "sector_index_map.csv"
NIFTY100_PATH = ROOT / "data" / "nifty100_symbols.txt"

# yfinance suffix for NSE
YF_SUFFIX = ".NS"

# Download / training limits (full ~2000+ liquid names; set lower for laptops)
MAX_SYMBOLS = int(os.environ.get("MAX_SYMBOLS", "500"))
DOWNLOAD_WORKERS = int(os.environ.get("DOWNLOAD_WORKERS", "8"))

# Circuit filter: flag days with extreme close-to-close moves (approximate; tune by segment)
CIRCUIT_RETURN_THRESHOLD = float(os.environ.get("CIRCUIT_RETURN_THRESHOLD", "0.19"))

# Features
LOOKBACK = 60
WALK_FORWARD_TEST_YEARS = 1

# Model
LSTM_UNITS = 64
TRANSFORMER_HEADS = 4
TRANSFORMER_FF = 128
DROPOUT = 0.15
EPOCHS_GLOBAL = int(os.environ.get("EPOCHS_GLOBAL", "40"))
EPOCHS_FINETUNE = int(os.environ.get("EPOCHS_FINETUNE", "25"))
BATCH_SIZE = 64

# Scanner (IST)
SCAN_HOUR = 15
SCAN_MINUTE = 45
TZ_IST = "Asia/Kolkata"

# Telegram
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

# VaR
VAR_CONFIDENCE = 0.95
VAR_LOOKBACK_DAYS = 252
