"""
Z-score returns, sector relative strength (alpha vs sector index), circuit-day filter.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import CIRCUIT_RETURN_THRESHOLD

logger = logging.getLogger(__name__)


def load_sector_map(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def fetch_sector_index_returns(
    index_tickers: list[str],
    period: str = "10y",
) -> dict[str, pd.Series]:
    """Daily log returns for each sector index."""
    out: dict[str, pd.Series] = {}
    for ix in sorted(set(index_tickers)):
        try:
            d = yf.Ticker(ix).history(period=period, interval="1d", auto_adjust=True)
            if d.empty:
                continue
            lc = np.log(d["Close"].clip(lower=1e-12))
            out[ix] = lc.diff()
        except Exception as e:
            logger.warning("Sector index %s: %s", ix, e)
    return out


def attach_sector_alpha(
    df: pd.DataFrame,
    sector_map: pd.DataFrame,
    sector_returns: dict[str, pd.Series],
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """sector_alpha = stock log return minus sector index log return (aligned by date)."""
    out = df.copy()
    out["sym_root"] = out[symbol_col].str.replace(".NS", "", regex=False)
    sm = sector_map.drop_duplicates("symbol_root")
    out = out.merge(sm, left_on="sym_root", right_on="symbol_root", how="left")

    def _one(sym_df: pd.DataFrame) -> pd.DataFrame:
        sym_df = sym_df.sort_index()
        ix = sym_df["yahoo_sector_index"].iloc[0]
        sret = np.log(sym_df["close"] / sym_df["close"].shift(1))
        if pd.isna(ix) or ix not in sector_returns:
            # Neutral alpha when symbol has no sector row or Yahoo index fetch failed
            sym_df["sector_alpha"] = 0.0
            return sym_df
        bret = sector_returns[ix].reindex(sym_df.index).ffill().bfill()
        sym_df["sector_alpha"] = sret.values - bret.values
        return sym_df

    parts = []
    for sym, g in out.groupby(symbol_col, sort=False):
        parts.append(_one(g))
    merged = pd.concat(parts, axis=0).sort_index()
    return merged.drop(columns=["sym_root", "symbol_root"], errors="ignore")


def zscore_returns(close: pd.Series, window: int = 60) -> pd.Series:
    r = np.log(close / close.shift(1))
    mu = r.rolling(window, min_periods=20).mean()
    sd = r.rolling(window, min_periods=20).std()
    return (r - mu) / sd.clip(lower=1e-8)


def flag_circuit_days(close: pd.Series, threshold: float = CIRCUIT_RETURN_THRESHOLD) -> pd.Series:
    ret = close.pct_change().abs()
    return (ret >= threshold).astype(np.float32)


def enrich_symbol_frame(
    df: pd.DataFrame,
    sector_map: pd.DataFrame,
    sector_returns: dict[str, pd.Series],
) -> pd.DataFrame:
    g = df.sort_index()
    g["z_ret"] = zscore_returns(g["close"])
    g["circuit_flag"] = flag_circuit_days(g["close"])
    if "delivery_pct" not in g.columns:
        g["delivery_pct"] = np.nan
    g["delivery_pct"] = g["delivery_pct"].ffill().fillna(0.0)
    g["log_vol"] = np.log(g["volume"].clip(lower=1) + 1.0)
    g = attach_sector_alpha(g, sector_map, sector_returns)
    g["target"] = np.log(g["close"].shift(-1) / g["close"])
    return g
