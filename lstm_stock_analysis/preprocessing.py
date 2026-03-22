"""MinMax scaling and chronological train/test split with LSTM sequences."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from config import LOOKBACK, TRAIN_RATIO
from features import CLOSE_FEATURE_INDEX, FEATURE_COLUMNS


def build_xy_sequences(
    scaled: np.ndarray,
    lookback: int,
    split_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    `scaled`: (n_timesteps, n_features), scaled full series (train fit scaler applied to all).

    Train samples: indices i in [lookback, split_idx) — predict close at same row i.
    Test samples: indices i in [split_idx, n) — sequences may use rows before split_idx.

    Returns X_train, y_train, X_test, y_test.
    """
    n = scaled.shape[0]
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    train_mask: list[bool] = []

    for i in range(lookback, n):
        X_list.append(scaled[i - lookback : i])
        y_list.append(scaled[i, CLOSE_FEATURE_INDEX])
        train_mask.append(i < split_idx)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)

    train_mask = np.array(train_mask, dtype=bool)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]
    return X_train, y_train, X_test, y_test


def prepare_data_for_ticker(
    feature_df,
    lookback: int = LOOKBACK,
    train_ratio: float = TRAIN_RATIO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, int, np.ndarray]:
    """
    Drop NaNs, fit MinMaxScaler on train rows only, transform full matrix,
    split index at train_ratio, build sequences.

    Returns X_train, y_train, X_test, y_test, scaler, split_idx,
    close_all (aligned raw close prices for the cleaned rows).
    """
    data = feature_df[FEATURE_COLUMNS].copy()
    data = data.dropna()
    values = data.values.astype(np.float64)
    close_all = data["Adj Close"].values.astype(np.float64)
    n = len(values)
    split_idx = int(n * train_ratio)
    if split_idx <= lookback:
        raise ValueError(
            f"Train split index {split_idx} must exceed lookback {lookback}. "
            "Use more history or reduce LOOKBACK."
        )
    if n - split_idx < 2:
        raise ValueError(
            f"Not enough test rows (n={n}, split_idx={split_idx})."
        )

    train_vals = values[:split_idx]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_vals)
    scaled = scaler.transform(values)

    X_train, y_train, X_test, y_test = build_xy_sequences(scaled, lookback, split_idx)
    return X_train, y_train, X_test, y_test, scaler, split_idx, close_all


def inverse_transform_close(
    scaler: MinMaxScaler,
    close_scaled: np.ndarray,
) -> np.ndarray:
    """Map scaled close predictions back to price using a dummy full row."""
    close_scaled = np.asarray(close_scaled, dtype=np.float64).reshape(-1)
    n_feat = len(scaler.scale_)
    pad = np.zeros((len(close_scaled), n_feat), dtype=np.float64)
    pad[:, CLOSE_FEATURE_INDEX] = close_scaled
    inv = scaler.inverse_transform(pad)
    return inv[:, CLOSE_FEATURE_INDEX]
