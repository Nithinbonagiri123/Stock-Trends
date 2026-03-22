"""
Technical indicators (pandas_ta), lags, volume ROC, and next-day return target.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


class FeatureEngineeringError(RuntimeError):
    """Invalid input or failed feature build."""


class FeatureEngineering:
    """Augment merged OHLCV + context with TA columns and supervised-learning target."""

    def __init__(self, price_columns: Sequence[str] | None = None) -> None:
        self.price_columns = price_columns

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expects columns at minimum: Open, High, Low, Close, Volume,
        plus any *_ctx context columns.

        Adds: RSI, MACD stack, Bollinger Bands, ATR, Close lags, volume ROC,
        and Target = next-day fractional return of Close.
        """
        if df.empty:
            raise FeatureEngineeringError("Empty DataFrame.")

        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise FeatureEngineeringError(f"Missing columns: {missing}")

        out = df.copy()

        try:
            # RSI (14) — explicit series API for stable column naming
            out["RSI_14"] = ta.rsi(out["Close"], length=14)

            macd = ta.macd(out["Close"], fast=12, slow=26, signal=9)
            if macd is None or macd.empty:
                raise FeatureEngineeringError("MACD returned empty.")
            out = out.join(macd, how="left")

            bb = ta.bbands(out["Close"], length=20, std=2.0)
            if bb is None or bb.empty:
                raise FeatureEngineeringError("Bollinger bands returned empty.")
            out = out.join(bb, how="left")

            out["ATR_14"] = ta.atr(
                high=out["High"],
                low=out["Low"],
                close=out["Close"],
                length=14,
            )
        except FeatureEngineeringError:
            raise
        except Exception as e:
            raise FeatureEngineeringError(f"pandas_ta indicator failed: {e}") from e

        for lag in (1, 2, 3, 5):
            out[f"Close_lag_{lag}"] = out["Close"].shift(lag)

        out["Volume_roc_5"] = out["Volume"].pct_change(periods=5)

        out["Target"] = out["Close"].shift(-1) / out["Close"] - 1.0

        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna(subset=["Target"])
        if out.empty:
            raise FeatureEngineeringError("No rows left after building target.")

        out = out.dropna()
        if len(out) < 100:
            raise FeatureEngineeringError("Insufficient rows after feature engineering.")

        logger.info("Feature matrix: %s rows, %s columns.", len(out), out.shape[1])
        return out

    @staticmethod
    def feature_columns_for_model(df: pd.DataFrame) -> list[str]:
        """All numeric columns except the target (and non-feature id columns)."""
        exclude = {"Target"}
        cols = []
        for c in df.columns:
            if c in exclude:
                continue
            if df[c].dtype.kind not in ("f", "i", "u"):
                continue
            cols.append(c)
        if not cols:
            raise FeatureEngineeringError("No numeric feature columns.")
        return cols
