"""Rule-based buy signals from predicted vs current close."""

from __future__ import annotations

import numpy as np

from config import BUY_THRESHOLD


def buy_signals(
    current_close: np.ndarray,
    predicted_next_close: np.ndarray,
    threshold: float = BUY_THRESHOLD,
) -> np.ndarray:
    """
    For each day t, compare predicted price for t+1 to current close at t.

    Buy signal if predicted_next_close > current_close * (1 + threshold).

    Returns boolean array aligned with `current_close`.
    """
    current_close = np.asarray(current_close, dtype=np.float64)
    predicted_next_close = np.asarray(predicted_next_close, dtype=np.float64)
    if current_close.shape != predicted_next_close.shape:
        raise ValueError("current_close and predicted_next_close must have the same shape.")
    return predicted_next_close > current_close * (1.0 + threshold)


def summarize_signals(signals: np.ndarray) -> dict[str, int | float]:
    n = len(signals)
    buys = int(np.sum(signals))
    return {
        "n_days": n,
        "buy_count": buys,
        "buy_fraction": float(buys / n) if n else 0.0,
    }
