"""Automation-specific settings (env overridable)."""

from __future__ import annotations

import os
from pathlib import Path

# Paths (under quant_system/automation/)
AUTOMATION_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = AUTOMATION_ROOT / "artifacts"
DB_PATH = AUTOMATION_ROOT / "prediction_log.sqlite"

MODEL_FILENAME = "model.keras"
SCALER_FILENAME = "scaler.pkl"
FEATURE_COLUMNS_FILENAME = "feature_columns.json"
META_FILENAME = "meta.json"

# Scheduling (US equities, America/New_York)
SCHEDULE_TIMEZONE = os.environ.get("SCHEDULE_TIMEZONE", "America/New_York")
# Default: 15:00 = 3pm ET ≈ 1 hour before 4pm close
SCHEDULE_HOUR = int(os.environ.get("SCHEDULE_HOUR", "15"))
SCHEDULE_MINUTE = int(os.environ.get("SCHEDULE_MINUTE", "0"))

# Retries
RETRY_DELAY_SECONDS = int(os.environ.get("RETRY_DELAY_SECONDS", "300"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))

# Target symbol for the automated job (override with PIPELINE_TICKER)
PIPELINE_TICKER = os.environ.get("PIPELINE_TICKER", "AAPL").strip().upper()

# Alerts
BUY_THRESHOLD = float(os.environ.get("BUY_THRESHOLD", "0.015"))
MAX_RSI_ALERT = float(os.environ.get("MAX_RSI_ALERT", "70"))

# Telegram (optional)
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

# Email (optional)
EMAIL_SMTP_HOST = os.environ.get("EMAIL_SMTP_HOST", "").strip()
EMAIL_SMTP_PORT = int(os.environ.get("EMAIL_SMTP_PORT", "587"))
EMAIL_SMTP_USER = os.environ.get("EMAIL_SMTP_USER", "").strip()
EMAIL_SMTP_PASSWORD = os.environ.get("EMAIL_SMTP_PASSWORD", "").strip()
EMAIL_FROM = os.environ.get("EMAIL_FROM", "").strip()
EMAIL_TO = os.environ.get("EMAIL_TO", "").strip()
