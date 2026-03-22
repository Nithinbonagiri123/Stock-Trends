"""
Download target equity + market context (S&P 500, VIX, 10Y yield) and merge into one frame.
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import pandas as pd
import yfinance as yf

from config import CONTEXT_SYMBOLS, DEFAULT_TICKER, INTERVAL, YEARS_HISTORY

logger = logging.getLogger(__name__)


def _ts_to_naive_ny_date(ts: pd.Timestamp) -> pd.Timestamp:
    """Use NY calendar date for daily bars (Yahoo mixes UTC / America/New_York)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("America/New_York")
    return pd.Timestamp(t.date())


def _normalize_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([_ts_to_naive_ny_date(pd.Timestamp(x)) for x in idx])


def _align_context_to_target(
    ctx_series: pd.Series,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Reindex context (VIX, etc.) onto equity dates. Yahoo indices often differ in
    timezone; naive reindex can yield all-NaN without normalization.
    """
    s = ctx_series.copy()
    s.index = _normalize_index(pd.DatetimeIndex(s.index))
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    tgt = _normalize_index(pd.DatetimeIndex(target_index))
    out = s.reindex(tgt).ffill().bfill()
    if out.isna().any():
        out = out.ffill().bfill()
    if out.isna().any() and s.notna().any():
        # Rare: no date overlap — broadcast last known value
        out = out.fillna(s.dropna().iloc[-1])
    return out


class DataIngestionError(RuntimeError):
    """Raised when downloads fail or produce unusable data."""


class DataIngestion:
    """
    Pull daily history for a target ticker and context series, merge on calendar,
    forward-fill missing context values (e.g. misaligned holidays).
    """

    CONTEXT_ALIASES = {"^GSPC": "SPX", "^VIX": "VIX", "^TNX": "TNX_10Y"}

    def __init__(
        self,
        ticker: str = DEFAULT_TICKER,
        years: int = YEARS_HISTORY,
        context_symbols: Sequence[str] = CONTEXT_SYMBOLS,
    ) -> None:
        self.ticker = ticker.strip().upper()
        self.years = int(years)
        self.context_symbols = tuple(context_symbols)

    def _download_symbol(self, symbol: str) -> pd.DataFrame:
        try:
            t = yf.Ticker(symbol)
            df = t.history(period=f"{self.years}y", interval=INTERVAL, auto_adjust=False)
        except Exception as e:
            raise DataIngestionError(f"Failed to download {symbol}: {e}") from e

        if df is None or df.empty:
            raise DataIngestionError(f"No rows returned for {symbol}.")
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    def download_raw_target(self) -> pd.DataFrame:
        """OHLCV (+ Adj Close) for the primary ticker."""
        df = self._download_symbol(self.ticker)
        return df

    def build_master_frame(self) -> pd.DataFrame:
        """
        Align target OHLCV to its native calendar; reindex each context series onto
        those dates and forward-fill so macro series follow the equity sessions.
        """
        main = self.download_raw_target()
        if main.empty:
            raise DataIngestionError(f"Target {self.ticker} has no data.")

        idx = main.index.sort_values()
        master = main.reindex(idx).copy()

        for sym in self.context_symbols:
            ctx = self._download_symbol(sym)
            alias = self.CONTEXT_ALIASES.get(sym, sym.replace("^", "").replace(".", "_"))
            price_col = "Adj Close" if "Adj Close" in ctx.columns else "Close"
            if price_col not in ctx.columns:
                raise DataIngestionError(f"{sym} missing Close/Adj Close.")
            series = _align_context_to_target(ctx[price_col], idx)
            # Use equity index labels so timezone-aware main aligns with context values
            series = pd.Series(series.values, index=idx)
            if series.isna().all():
                raise DataIngestionError(
                    f"Could not align {sym} to {self.ticker} calendar (no overlapping data).",
                )
            if series.isna().any():
                med = float(series.median()) if series.notna().any() else float("nan")
                if not math.isnan(med):
                    series = series.fillna(med)
                else:
                    series = series.ffill().bfill()
                if series.isna().any():
                    logger.warning(
                        "%s context still has %s NaN after fill (check data source).",
                        sym,
                        int(series.isna().sum()),
                    )
            master[f"{alias}_ctx"] = series

        master = master.dropna(subset=["Open", "High", "Low", "Close"])

        if master.empty:
            raise DataIngestionError("Master frame empty after merge.")

        logger.info(
            "Master frame: %s rows from %s to %s",
            len(master),
            master.index.min().date(),
            master.index.max().date(),
        )
        return master
