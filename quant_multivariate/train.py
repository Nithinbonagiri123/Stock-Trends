"""
Walk-forward validation, label smoothing, ReduceLROnPlateau, Sharpe-based checkpoint.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    EPOCHS_PER_FOLD,
    LABEL_SMOOTHING_ALPHA,
    LOOKBACK,
    LR_FACTOR,
    LR_MIN,
    LR_PATIENCE,
    MIN_TRAIN_YEARS,
    MODEL_NAME,
    RANDOM_SEED,
    EARLY_STOP_PATIENCE,
)

from model import build_model

logger = logging.getLogger(__name__)


def _sequences(X: np.ndarray, y: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    n = len(X)
    for i in range(lookback, n):
        xs.append(X[i - lookback : i])
        ys.append(y[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        return float("nan")
    sd = np.std(r, ddof=1)
    if sd < 1e-12:
        return float("nan")
    return float(np.mean(r) / sd * np.sqrt(252.0))


class SharpeCheckpoint(tf.keras.callbacks.Callback):
    """Save weights only when validation Sharpe improves."""

    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, path: Path):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.path = path
        self.best_sharpe = -np.inf

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        pred = self.model.predict(self.X_val, verbose=0).flatten()
        strat = np.sign(pred) * self.y_val
        sh = _sharpe(strat)
        if not np.isnan(sh) and sh > self.best_sharpe:
            self.best_sharpe = sh
            self.model.save(self.path, overwrite=True)
            logger.info("Epoch %s: new best val Sharpe=%.4f → saved %s", epoch + 1, sh, self.path)


def walk_forward_years(dates: pd.DatetimeIndex) -> list[tuple[pd.Series, pd.Series, int]]:
    years = sorted(pd.unique(dates.year))
    out: list[tuple[pd.Series, pd.Series, int]] = []
    for test_y in years:
        train_years = [y for y in years if y < test_y]
        if len(train_years) < MIN_TRAIN_YEARS:
            continue
        train_mask = pd.Series(dates.year < test_y, index=dates)
        test_mask = pd.Series(dates.year == test_y, index=dates)
        out.append((train_mask, test_mask, test_y))
    return out


def _row_to_seq_idx(i: int, lookback: int) -> int | None:
    if i < lookback:
        return None
    return i - lookback


def run_walk_forward(feat_df: pd.DataFrame, y_raw: pd.Series) -> dict:
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    dates = feat_df.index
    X_all = feat_df.values.astype(np.float64)
    y_all = y_raw.values.astype(np.float64)
    n = len(feat_df)
    n_feat = feat_df.shape[1]

    folds = walk_forward_years(dates)
    if not folds:
        raise RuntimeError("Not enough years for walk-forward; lower MIN_TRAIN_YEARS.")

    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    best_path = Path(ARTIFACTS_DIR) / MODEL_NAME

    fold_results: list[dict] = []

    for train_mask, test_mask, test_y in folds:
        logger.info("=== Walk-forward fold: test year %s ===", test_y)

        train_ix = np.where(train_mask.values)[0]
        test_ix = np.where(test_mask.values)[0]
        train_ix = train_ix[train_ix >= LOOKBACK]
        test_ix = test_ix[test_ix >= LOOKBACK]
        if len(train_ix) < 100 or len(test_ix) < 5:
            logger.warning("Skipping fold %s: insufficient rows after warmup.", test_y)
            continue

        split = max(int(len(train_ix) * 0.85), 1)
        tr_ix = train_ix[:split]
        va_ix = train_ix[split:]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X_all[train_mask.values])
        X_scaled = scaler.transform(X_all)

        tr_mean = float(np.mean(y_all[train_mask.values]))
        y_smooth = y_all.copy()
        for i in np.concatenate([tr_ix, va_ix]):
            y_smooth[i] = (1.0 - LABEL_SMOOTHING_ALPHA) * y_all[i] + LABEL_SMOOTHING_ALPHA * tr_mean

        X_seq, y_seq_s = _sequences(X_scaled, y_smooth, LOOKBACK)
        _, y_seq_raw = _sequences(X_scaled, y_all, LOOKBACK)

        def seq_pick(row_ix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            sidx = []
            for i in row_ix:
                k = _row_to_seq_idx(i, LOOKBACK)
                if k is not None:
                    sidx.append(k)
            sidx = sorted(set(sidx))
            if not sidx:
                return (
                    np.empty((0, LOOKBACK, n_feat)),
                    np.empty((0,)),
                    np.empty((0,)),
                )
            return X_seq[sidx], y_seq_s[sidx], y_seq_raw[sidx]

        X_tr, y_tr, _ = seq_pick(tr_ix)
        X_va, y_va, _ = seq_pick(va_ix)
        X_te, _, y_te = seq_pick(test_ix)

        if len(X_tr) < 32 or len(X_va) < 8 or len(X_te) < 3:
            logger.warning("Skipping fold %s: sequence batches too small.", test_y)
            continue

        model = build_model(LOOKBACK, n_feat)

        callbacks = [
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=LR_FACTOR,
                patience=LR_PATIENCE,
                min_lr=LR_MIN,
                verbose=1,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOP_PATIENCE,
                restore_best_weights=False,
                verbose=1,
            ),
            SharpeCheckpoint(X_va, y_va, best_path),
        ]

        model.fit(
            X_tr,
            y_tr,
            validation_data=(X_va, y_va),
            epochs=EPOCHS_PER_FOLD,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )

        if best_path.exists():
            model = load_model(best_path)
        else:
            logger.warning("No Sharpe checkpoint; using last epoch weights for test.")

        pred_te = model.predict(X_te, verbose=0).flatten()
        strat = np.sign(pred_te) * y_te
        sh_te = _sharpe(strat)
        mse = float(np.mean((pred_te - y_te) ** 2))

        fold_results.append(
            {
                "test_year": test_y,
                "test_sharpe": sh_te,
                "test_mse": mse,
                "n_test": len(y_te),
            },
        )
        logger.info("Fold year %s — test Sharpe=%.4f MSE=%.6f (n=%s)", test_y, sh_te, mse, len(y_te))

    return {"folds": fold_results, "model_path": str(best_path)}


def load_and_train(panel: pd.DataFrame) -> dict:
    from features import build_feature_matrix

    feat_df, y_raw = build_feature_matrix(panel)
    return run_walk_forward(feat_df, y_raw)
