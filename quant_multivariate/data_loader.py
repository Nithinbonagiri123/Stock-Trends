"""Download and align multivariate daily series from yfinance."""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def download_aligned(symbols: tuple[str, ...], years: int = 10) -> pd.DataFrame:
    """Single frame: MultiIndex columns (symbol, field) or flat names per symbol."""
    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        t = yf.Ticker(sym)
        df = t.history(period=f"{years}y", interval="1d", auto_adjust=True)
        if df.empty:
            raise RuntimeError(f"No data for {sym}")
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        # Standardize column names
        rename = {}
        if "Close" in df.columns:
            rename["Close"] = "close"
        if "Volume" in df.columns:
            rename["Volume"] = "volume"
        df = df.rename(columns=rename)
        if "close" not in df.columns:
            raise RuntimeError(f"{sym}: missing Close")
        frames[sym] = df[["close", "volume"]] if "volume" in df.columns else df[["close"]]

    # Align on union calendar, forward-fill (macro/metal holidays differ)
    idx = None
    for sym, df in frames.items():
        idx = df.index if idx is None else idx.union(df.index)
    idx = idx.sort_values()

    out = pd.DataFrame(index=idx)
    for sym, df in frames.items():
        for col in df.columns:
            out[f"{sym}__{col}"] = df[col].reindex(idx).ffill().bfill()

    out = out.dropna(how="any")
    logger.info("Aligned panel: %s rows, %s → %s", len(out), out.index.min(), out.index.max())
    return out
