"""
Simulate long-only next-day trades from model predictions with RSI risk filter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from config import MAX_RSI_FOR_BUY, MIN_PREDICTED_GAIN

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    sharpe_ratio: float
    max_drawdown: float
    cumulative_return: float
    hit_rate: float
    n_signals: int
    n_steps: int


class Backtester:
    """
    BUY (take next-day exposure) only if:
      - predicted next-day return > MIN_PREDICTED_GAIN (1.5%), and
      - RSI < MAX_RSI_FOR_BUY (70).

    Strategy PnL on a step: actual next-day return if BUY else 0 (cash).
    """

    def __init__(
        self,
        min_predicted_gain: float = MIN_PREDICTED_GAIN,
        max_rsi_for_buy: float = MAX_RSI_FOR_BUY,
    ) -> None:
        self.min_predicted_gain = float(min_predicted_gain)
        self.max_rsi_for_buy = float(max_rsi_for_buy)

    def generate_signals(
        self,
        predicted_returns: np.ndarray,
        rsi: np.ndarray,
    ) -> np.ndarray:
        pred = np.asarray(predicted_returns, dtype=np.float64)
        rsi_arr = np.asarray(rsi, dtype=np.float64)

        if pred.shape != rsi_arr.shape:
            raise ValueError("predicted_returns and rsi must have the same shape.")

        # If RSI missing (zeros), only apply gain rule (documented behavior)
        if np.allclose(rsi_arr, 0.0):
            logger.warning(
                "RSI is all zeros — applying gain filter only (no RSI gate).",
            )
            return pred > self.min_predicted_gain

        gain_ok = pred > self.min_predicted_gain
        rsi_ok = rsi_arr < self.max_rsi_for_buy
        return gain_ok & rsi_ok

    @staticmethod
    def _sharpe_daily(returns: np.ndarray, trading_days: float = 252.0) -> float:
        r = np.asarray(returns, dtype=np.float64)
        if len(r) < 2:
            return float("nan")
        mu = np.mean(r)
        sd = np.std(r, ddof=1)
        if sd < 1e-12:
            return float("nan")
        return float((mu / sd) * np.sqrt(trading_days))

    @staticmethod
    def _max_drawdown(equity_curve: np.ndarray) -> float:
        """Return most negative peak-to-trough drawdown (e.g. -0.25 for -25%)."""
        eq = np.asarray(equity_curve, dtype=np.float64)
        if len(eq) < 2:
            return float("nan")
        peak = np.maximum.accumulate(eq)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = (eq - peak) / peak
        return float(np.nanmin(dd))

    def run(
        self,
        predicted_returns: np.ndarray,
        actual_returns: np.ndarray,
        rsi: np.ndarray,
    ) -> tuple[BacktestResult, dict[str, np.ndarray]]:
        """
        Simulate step-by-step: if signal, earn `actual_returns`; else 0.

        Returns BacktestResult plus a dict with equity_curve, strategy_returns, signals.
        """
        pred = np.asarray(predicted_returns, dtype=np.float64).flatten()
        act = np.asarray(actual_returns, dtype=np.float64).flatten()
        rsi_arr = np.asarray(rsi, dtype=np.float64).flatten()

        if not (len(pred) == len(act) == len(rsi_arr)):
            raise ValueError("predicted_returns, actual_returns, rsi must align in length.")

        signals = self.generate_signals(pred, rsi_arr)
        strat_rets = np.where(signals, act, 0.0)
        equity = np.cumprod(1.0 + strat_rets)

        sharpe = self._sharpe_daily(strat_rets)
        mdd = self._max_drawdown(equity)
        cum_ret = float(equity[-1] - 1.0) if len(equity) else 0.0

        # Hit rate: when signal, did actual return > 0
        if np.any(signals):
            hit = float(np.mean(act[signals] > 0))
        else:
            hit = float("nan")

        result = BacktestResult(
            sharpe_ratio=sharpe,
            max_drawdown=mdd,
            cumulative_return=cum_ret,
            hit_rate=hit,
            n_signals=int(np.sum(signals)),
            n_steps=len(strat_rets),
        )

        extra = {
            "equity_curve": equity,
            "strategy_returns": strat_rets,
            "signals": signals.astype(np.int8),
        }
        return result, extra
