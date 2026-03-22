"""
Log returns, fractional differentiation, Fourier cycle features, and pandas_ta overlays.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pandas_ta as ta

from config import (
    FOURIER_TOP_K,
    FRAC_DIFF_D,
    FRAC_DIFF_WIDTH,
    SYMBOLS,
    TARGET_SYMBOL,
)
from fracdiff import fractional_diff_ffd

logger = logging.getLogger(__name__)


def _log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def fourier_cycle_features(log_btc: np.ndarray, top_k: int) -> np.ndarray:
    """
    Dominant cycles from FFT of detrended log BTC; sin/cos features per timestep.
    """
    n = len(log_btc)
    if n < 32:
        return np.zeros((n, top_k * 2))
    t = np.arange(n, dtype=np.float64)
    detrended = log_btc - np.linspace(log_btc[0], log_btc[-1], n)
    spec = np.fft.rfft(detrended)
    mag = np.abs(spec)
    # exclude DC
    if len(mag) <= 2:
        return np.zeros((n, top_k * 2))
    order = np.argsort(mag[1:])[-top_k:][::-1] + 1
    feats = np.zeros((n, top_k * 2), dtype=np.float64)
    for i, pk in enumerate(order):
        period = float(n) / max(float(pk), 1.0)
        period = max(period, 2.0)
        feats[:, 2 * i] = np.sin(2.0 * np.pi * t / period)
        feats[:, 2 * i + 1] = np.cos(2.0 * np.pi * t / period)
    return feats


def build_feature_matrix(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Multivariate features for all symbols; target = next-day log return of TARGET_SYMBOL (BTC).
    """
    cols: dict[str, pd.Series] = {}
    btc_close = panel[f"{TARGET_SYMBOL}__close"].astype(np.float64)

    for sym in SYMBOLS:
        close = panel[f"{sym}__close"].astype(np.float64)
        lp = np.log(close.clip(lower=1e-12))
        cols[f"{sym}__log_close"] = lp
        cols[f"{sym}__log_ret"] = _log_returns(close)
        cols[f"{sym}__frac_diff_logp"] = fractional_diff_ffd(lp, FRAC_DIFF_D, FRAC_DIFF_WIDTH)
        rsi = ta.rsi(close, length=14)
        if rsi is None:
            rsi = pd.Series(np.nan, index=close.index)
        cols[f"{sym}__rsi14"] = rsi
        if f"{sym}__volume" in panel.columns:
            vol = panel[f"{sym}__volume"].astype(np.float64)
            cols[f"{sym}__vol_roc5"] = vol.pct_change(5)
        else:
            cols[f"{sym}__vol_roc5"] = pd.Series(0.0, index=close.index)

    log_btc = np.log(btc_close.clip(lower=1e-12)).values
    fourier = fourier_cycle_features(log_btc, FOURIER_TOP_K)
    for j in range(fourier.shape[1]):
        cols[f"fourier_{j}"] = pd.Series(fourier[:, j], index=panel.index)

    feat_df = pd.DataFrame(cols, index=panel.index)
    target = _log_returns(btc_close).shift(-1)
    feat_df["__target_raw"] = target

    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna()
    y = feat_df.pop("__target_raw")
    return feat_df, y
