"""
Merge NSE Bhavcopy **delivery** quantity / % when available.

NSE publishes daily files; URL patterns change. This module tries a known archive pattern
and falls back to NaN columns so the pipeline still runs offline.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def fetch_bhavcopy_delivery(date: datetime | None = None) -> pd.DataFrame | None:
    """
    Returns DataFrame with columns: symbol, delivery_qty, traded_qty, delivery_pct (if derivable).
    `date` defaults to last completed session (approximate).
    """
    if date is None:
        date = datetime.utcnow() - timedelta(days=1)
    # Example pattern (may require session cookies in production — use NSE official API / vendor)
    url = date.strftime(
        "https://nsearchives.nseindia.com/content/historical/EQUITIES/%Y/%b/cm%dd%b%Ybhav.csv"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)",
        "Accept": "text/csv",
    }
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code != 200:
            logger.warning("Bhavcopy HTTP %s for %s", r.status_code, url)
            return None
        raw = pd.read_csv(io.StringIO(r.text))
        # Normalize column names (NSE files vary)
        raw.columns = [c.strip().upper() for c in raw.columns]
        out = pd.DataFrame()
        if "SYMBOL" in raw.columns:
            out["symbol"] = raw["SYMBOL"].astype(str).str.upper()
        if "DELIV_QTY" in raw.columns:
            out["delivery_qty"] = pd.to_numeric(raw["DELIV_QTY"], errors="coerce")
        if "TOTTRDQTY" in raw.columns:
            out["traded_qty"] = pd.to_numeric(raw["TOTTRDQTY"], errors="coerce")
        if "delivery_qty" in out.columns and "traded_qty" in out.columns:
            out["delivery_pct"] = (out["delivery_qty"] / out["traded_qty"].clip(lower=1)) * 100.0
        return out
    except Exception as e:
        logger.warning("Bhavcopy fetch failed: %s", e)
        return None


def merge_delivery_into_daily(daily: pd.DataFrame, bhav: pd.DataFrame | None) -> pd.DataFrame:
    """Left-merge on symbol + date index."""
    out = daily.copy()
    out["date"] = pd.DatetimeIndex(out.index).normalize()
    out["sym"] = out["symbol"].str.replace(".NS", "", regex=False)
    if bhav is None or bhav.empty:
        out["delivery_pct"] = float("nan")
        return out.drop(columns=["sym"], errors="ignore")
    b = bhav.rename(columns={"symbol": "sym"})
    if "sym" not in b.columns and "SYMBOL" in b.columns:
        b["sym"] = b["SYMBOL"]
    merged = out.merge(b[["sym", "delivery_pct"]], on="sym", how="left", suffixes=("", "_bhav"))
    if "delivery_pct_bhav" in merged.columns:
        merged["delivery_pct"] = merged["delivery_pct_bhav"].combine_first(merged.get("delivery_pct"))
        merged = merged.drop(columns=["delivery_pct_bhav"], errors="ignore")
    return merged.drop(columns=["sym"], errors="ignore")
