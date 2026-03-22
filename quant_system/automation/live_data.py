"""
Merge 10-year daily history with today's intraday bar (yfinance) for seamless updates.
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from data_ingestion import DataIngestion

logger = logging.getLogger(__name__)


def _aggregate_intraday_to_daily(intra: pd.DataFrame) -> pd.Series | None:
    if intra is None or intra.empty:
        return None
    last_idx = intra.index[-1]
    ac = "Adj Close" if "Adj Close" in intra.columns else "Close"
    return pd.Series(
        {
            "Open": float(intra["Open"].iloc[0]),
            "High": float(intra["High"].max()),
            "Low": float(intra["Low"].min()),
            "Close": float(intra["Close"].iloc[-1]),
            "Adj Close": float(intra[ac].iloc[-1]),
            "Volume": float(intra["Volume"].sum()),
        },
        name=last_idx,
    )


def fetch_intraday_daily_bar(ticker: str, interval: str = "1m") -> pd.Series | None:
    """Today's session aggregated to one OHLCV row."""
    try:
        t = yf.Ticker(ticker)
        intra = t.history(period="1d", interval=interval, auto_adjust=False)
    except Exception as e:
        logger.warning("Intraday fetch failed for %s: %s", ticker, e)
        return None
    return _aggregate_intraday_to_daily(intra)


def fetch_context_latest_close(symbol: str) -> float | None:
    try:
        t = yf.Ticker(symbol)
        d = t.history(period="5d", interval="1d", auto_adjust=False)
        if d.empty:
            return None
        col = "Adj Close" if "Adj Close" in d.columns else "Close"
        return float(d[col].iloc[-1])
    except Exception as e:
        logger.warning("Context fetch failed for %s: %s", symbol, e)
        return None


def _normalize_daily_index(ts: pd.Timestamp) -> pd.Timestamp:
    """Match naive daily index used by yfinance history."""
    if ts.tzinfo is not None:
        ts = ts.tz_convert("America/New_York")
    return pd.Timestamp(ts.date())


def merge_today_intraday(master: pd.DataFrame, ticker: str, ingestion: DataIngestion) -> pd.DataFrame:
    """
    Replace or append today's row using 1m data for the target; refresh context columns
    on that row from latest daily closes.
    """
    out = master.sort_index().copy()
    bar = fetch_intraday_daily_bar(ticker, interval="1m")
    if bar is None:
        logger.warning("No intraday bar for %s; using daily master only.", ticker)
        return out

    day_key = _normalize_daily_index(pd.Timestamp(bar.name))
    # Align to existing index tz
    if isinstance(out.index, pd.DatetimeIndex) and len(out.index):
        sample = out.index[-1]
        if sample.tzinfo is None:
            day_key = day_key.tz_localize(None)
        else:
            day_key = day_key.tz_localize("America/New_York")

    new_vals = {
        "Open": bar["Open"],
        "High": bar["High"],
        "Low": bar["Low"],
        "Close": bar["Close"],
        "Adj Close": bar["Adj Close"],
        "Volume": bar["Volume"],
    }

    for sym in ingestion.context_symbols:
        alias = ingestion.CONTEXT_ALIASES[sym]
        col = f"{alias}_ctx"
        if col in out.columns:
            v = fetch_context_latest_close(sym)
            if v is not None:
                new_vals[col] = v

    if day_key not in out.index:
        row = pd.DataFrame([new_vals], index=[day_key])
        for c in out.columns:
            if c not in row.columns:
                row[c] = float("nan")
        row = row[out.columns]
        out = pd.concat([out, row]).sort_index()
        out = out[~out.index.duplicated(keep="last")]
    else:
        for k, v in new_vals.items():
            if k in out.columns:
                out.loc[day_key, k] = v

    return out.ffill()


def build_live_master(ticker: str, years: int = 10) -> pd.DataFrame:
    ing = DataIngestion(ticker=ticker, years=years)
    base = ing.build_master_frame()
    return merge_today_intraday(base, ticker, ing)
