"""Download daily OHLCV history via yfinance."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_ticker_history(
    ticker: str,
    years: int = 10,
) -> pd.DataFrame:
    """
    Download daily adjusted OHLCV for `ticker` over approximately `years` years.

    Returns a DataFrame indexed by date with columns:
    Open, High, Low, Close, Adj Close, Volume (yfinance defaults).
    """
    t = yf.Ticker(ticker)
    # Period-based download; yfinance resolves ~10y of daily bars
    df = t.history(period=f"{years}y", interval="1d", auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check symbol or network.")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def download_many(
    tickers: list[str],
    years: int = 10,
) -> dict[str, pd.DataFrame]:
    """Return mapping ticker -> DataFrame."""
    out: dict[str, pd.DataFrame] = {}
    for sym in tickers:
        out[sym] = download_ticker_history(sym, years=years)
    return out
