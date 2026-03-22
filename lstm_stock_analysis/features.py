"""Technical indicators: RSI(14), SMA(50/200), MACD."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing)."""
    delta = close.diff()
    gain = delta.where(delta > 0.0, 0.0)
    loss = (-delta).where(delta < 0.0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out


def add_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects yfinance columns including 'Close' and 'Adj Close'.
    Adds: rsi_14, sma_50, sma_200, macd_line, macd_signal, macd_hist.
    """
    out = df.copy()
    price = out["Adj Close"] if "Adj Close" in out.columns else out["Close"]

    out["rsi_14"] = rsi_wilder(price, 14)
    out["sma_50"] = price.rolling(window=50, min_periods=50).mean()
    out["sma_200"] = price.rolling(window=200, min_periods=200).mean()
    macd_line, macd_signal, macd_hist = add_macd(price)
    out["macd_line"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist

    return out


# Use adjusted close as the price series (aligned with indicator math)
FEATURE_COLUMNS = [
    "Adj Close",
    "rsi_14",
    "sma_50",
    "sma_200",
    "macd_line",
    "macd_signal",
    "macd_hist",
]

# Index of adjusted close inside FEATURE_COLUMNS (for inverse scaling / targets)
CLOSE_FEATURE_INDEX = FEATURE_COLUMNS.index("Adj Close")
