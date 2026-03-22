"""
Fixed-width fractional differentiation (FFD) for stationarity with memory preservation.
Reference: López de Prado — weights via binomial-style recurrence on window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ffd_weights(d: float, width: int) -> np.ndarray:
    """Weights for expanding window FFD; d in (0,1) typical."""
    if width < 2:
        width = 2
    w = np.zeros(width, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, width):
        w[k] = -w[k - 1] * (d - (k - 1)) / k
    return w[::-1]


def fractional_diff_ffd(series: pd.Series, d: float, width: int) -> pd.Series:
    """
    Apply fixed-width fractional differentiation along the series.
    Drops initial NaN warmup; aligns index with input.
    """
    s = series.astype(np.float64).values
    w = _ffd_weights(d, width)
    n = len(s)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(width - 1, n):
        out[i] = np.dot(w, s[i - width + 1 : i + 1])
    return pd.Series(out, index=series.index)
