"""
Historical VaR for top picks and simple sector concentration metrics.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import VAR_CONFIDENCE, VAR_LOOKBACK_DAYS
from data.features_engineering import load_sector_map

logger = logging.getLogger(__name__)


def _log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def historical_var_1d(
    returns: pd.Series,
    confidence: float = VAR_CONFIDENCE,
    lookback: int = VAR_LOOKBACK_DAYS,
) -> float:
    """Lower tail of return distribution (negative = loss). Uses last `lookback` days."""
    r = returns.dropna().tail(lookback)
    if len(r) < 30:
        return float("nan")
    alpha = 1.0 - confidence
    return float(np.percentile(r, alpha * 100.0))


def portfolio_equal_weight_var(
    panel: pd.DataFrame,
    symbols: list[str],
    confidence: float = VAR_CONFIDENCE,
    lookback: int = VAR_LOOKBACK_DAYS,
) -> float:
    """VaR of equal-weight portfolio of daily log returns (aligned intersection of dates)."""
    mats: list[pd.Series] = []
    for s in symbols:
        g = panel[panel["symbol"] == s].sort_index()
        if g.empty:
            continue
        mats.append(_log_returns(g["close"]).rename(s))
    if not mats:
        return float("nan")
    df = pd.concat(mats, axis=1).dropna(how="any")
    if df.empty:
        return float("nan")
    port = df.mean(axis=1).tail(lookback)
    return historical_var_1d(port, confidence=confidence, lookback=len(port))


def sector_concentration(
    symbols: list[str],
    sector_map_path: Path,
) -> pd.DataFrame:
    sm = load_sector_map(sector_map_path)
    roots = [s.replace(".NS", "").upper() for s in symbols]
    sub = sm[sm["symbol_root"].isin(roots)].copy()
    sub["weight"] = 1.0 / len(symbols)
    agg = sub.groupby("sector", dropna=False)["weight"].sum().reset_index()
    agg = agg.sort_values("weight", ascending=False)
    return agg


def risk_report_top_picks(
    panel: pd.DataFrame,
    top_symbols: list[str],
    sector_map_path: Path,
) -> dict:
    var_p = portfolio_equal_weight_var(panel, top_symbols)
    sectors = sector_concentration(top_symbols, sector_map_path)
    max_sector_share = float(sectors["weight"].max()) if len(sectors) else float("nan")
    return {
        "var_1d_equal_weight": var_p,
        "max_sector_weight_among_picks": max_sector_share,
        "sector_breakdown": sectors,
    }
