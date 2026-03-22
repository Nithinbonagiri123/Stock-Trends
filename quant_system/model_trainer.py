"""
LSTM + multi-head attention stack, MinMaxScaler, EarlyStopping.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model, load_model

from config import (
    ATTENTION_HEADS,
    ATTENTION_KEY_DIM,
    BATCH_SIZE,
    DROPOUT_RATE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    LOOKBACK,
    LSTM_UNITS,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
)

import pandas as pd

logger = logging.getLogger(__name__)


class ModelTrainerError(RuntimeError):
    """Model build / train failure."""


def _set_seed(seed: int) -> None:
    tf.random.set_seed(seed)
    np.random.seed(seed)


def _build_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal windows: rows [i-lookback, i) predict target[i]."""
    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    n = len(features)
    for i in range(lookback, n):
        x_list.append(features[i - lookback : i])
        y_list.append(targets[i])
    if not x_list:
        raise ModelTrainerError("No sequences; increase history or lower lookback.")
    return np.asarray(x_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)


def _temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> tuple[np.ndarray, ...]:
    m = len(X)
    i1 = max(1, int(m * train_ratio))
    i2 = max(i1 + 1, int(m * (train_ratio + val_ratio)))
    if i2 >= m:
        raise ModelTrainerError("Not enough sequences for train/val/test split.")
    return (
        X[:i1],
        y[:i1],
        X[i1:i2],
        y[i1:i2],
        X[i2:],
        y[i2:],
    )


def build_attention_lstm(
    lookback: int,
    n_features: int,
    lstm_units: tuple[int, int] = LSTM_UNITS,
    dropout: float = DROPOUT_RATE,
    num_heads: int = ATTENTION_HEADS,
    key_dim: int = ATTENTION_KEY_DIM,
) -> Model:
    """
    Two LSTM layers (return sequences on last for attention), Dropout, BatchNorm,
    MultiHeadAttention, pool, Dense(1) for predicted next-day return.
    """
    inp = Input(shape=(lookback, n_features))
    x = LSTM(lstm_units[0], return_sequences=True)(inp)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)
    x = LSTM(lstm_units[1], return_sequences=True)(x)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)

    attn = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout,
    )(x, x)
    x = tf.keras.layers.Add()([x, attn])
    x = GlobalAveragePooling1D()(x)
    out = Dense(1, activation="linear")(x)

    model = Model(inputs=inp, outputs=out, name="lstm_attention_regressor")
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model


class ModelTrainer:
    """End-to-end scaling, sequence build, training, and test predictions."""

    def __init__(
        self,
        lookback: int = LOOKBACK,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler: MinMaxScaler | None = None
        self.feature_names: list[str] | None = None
        self.model: Model | None = None

    def prepare_arrays(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Returns features matrix, targets, RSI array (for backtest), feature names."""
        try:
            rsi_col = next(c for c in df.columns if str(c).upper().startswith("RSI"))
        except StopIteration:
            rsi_col = None
            logger.warning("No RSI column found; backtest RSI filter will be skipped.")

        feats = df[feature_cols].values.astype(np.float64)
        targs = df["Target"].values.astype(np.float64)
        rsi = df[rsi_col].values.astype(np.float64) if rsi_col else np.zeros(len(df))

        n_train_rows = int(len(df) * TRAIN_RATIO)
        if n_train_rows <= self.lookback:
            raise ModelTrainerError("Train region too small for lookback window.")

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(feats[:n_train_rows])
        scaled = self.scaler.transform(feats)

        X, y = _build_sequences(scaled, targs, self.lookback)
        rsi_seq = rsi[self.lookback :]  # align with y (same length as X)

        self.feature_names = list(feature_cols)
        return X, y, rsi_seq, [rsi_col] if rsi_col else []

    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        rsi_aligned: np.ndarray,
    ) -> dict:
        """Train with early stopping; return test predictions and aligned RSI for backtest."""
        _set_seed(RANDOM_SEED)

        X_train, y_train, X_val, y_val, X_test, y_test = _temporal_split(
            X, y, TRAIN_RATIO, VAL_RATIO
        )
        m = len(rsi_aligned)
        i1 = max(1, int(m * TRAIN_RATIO))
        i2 = max(i1 + 1, int(m * (TRAIN_RATIO + VAL_RATIO)))
        if i2 >= m:
            raise ModelTrainerError("RSI array length mismatch for split.")
        rsi_test = rsi_aligned[i2:]

        n_feat = X_train.shape[2]
        self.model = build_attention_lstm(self.lookback, n_feat)

        es = EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )

        try:
            history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[es],
                verbose=1,
            )
        except Exception as e:
            raise ModelTrainerError(f"Training failed: {e}") from e

        pred_test = self.model.predict(X_test, verbose=0).flatten()

        return {
            "history": history,
            "y_test": y_test,
            "pred_test": pred_test,
            "rsi_test": rsi_test,
            "X_test": X_test,
        }

    def train_from_scratch(
        self,
        feat_df,
        feature_cols: list[str],
        artifact_dir: str | Path,
    ) -> dict:
        """Full train on engineered features and persist artifacts."""
        X, y, rsi, _ = self.prepare_arrays(feat_df, feature_cols)
        out = self.train_and_evaluate(X, y, rsi)
        self.save_artifacts(artifact_dir)
        return out

    def save_artifacts(self, directory: str | Path) -> None:
        """Persist Keras model, scaler, and feature column list for automated reload."""
        if self.model is None or self.scaler is None or not self.feature_names:
            raise ModelTrainerError("Nothing to save — train the model first.")
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        path_model = d / "model.keras"
        self.model.save(path_model)
        with (d / "scaler.pkl").open("wb") as f:
            pickle.dump(self.scaler, f)
        with (d / "feature_columns.json").open("w", encoding="utf-8") as f:
            json.dump(self.feature_names, f)
        meta = {
            "lookback": self.lookback,
            "keras_path": path_model.name,
        }
        with (d / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info("Saved artifacts to %s", d)

    def load_artifacts(self, directory: str | Path) -> None:
        """Load model + scaler + feature names (no training)."""
        d = Path(directory)
        with (d / "feature_columns.json").open(encoding="utf-8") as f:
            self.feature_names = json.load(f)
        with (d / "scaler.pkl").open("rb") as f:
            self.scaler = pickle.load(f)
        self.model = load_model(d / "model.keras")
        meta_path = d / "meta.json"
        if meta_path.exists():
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
            self.lookback = int(meta.get("lookback", self.lookback))
        logger.info("Loaded artifacts from %s", d)

    def predict_next(
        self,
        feat_df,
    ) -> dict:
        """
        One-step-ahead return prediction using the last `lookback` rows.
        Expects `feat_df` to contain all `feature_names` columns (engineered).
        """
        if self.model is None or self.scaler is None or not self.feature_names:
            raise ModelTrainerError("Model not loaded.")
        cols = self.feature_names
        missing = [c for c in cols if c not in feat_df.columns]
        if missing:
            raise ModelTrainerError(f"Missing feature columns: {missing}")

        feats = feat_df[cols].values.astype(np.float64)
        if len(feats) < self.lookback:
            raise ModelTrainerError(f"Need at least {self.lookback} rows; got {len(feats)}.")

        scaled = self.scaler.transform(feats)
        x = scaled[-self.lookback :].reshape(1, self.lookback, len(cols)).astype(np.float32)
        pred_ret = float(self.model.predict(x, verbose=0)[0, 0])

        last = feat_df.iloc[-1]
        try:
            rsi_col = next(c for c in feat_df.columns if str(c).upper().startswith("RSI"))
            rsi = float(last[rsi_col])
        except StopIteration:
            rsi = float("nan")

        close = float(last.get("Adj Close", last.get("Close", float("nan"))))
        pred_price = close * (1.0 + pred_ret)
        # Heuristic confidence: stronger magnitude => higher score (0–100), not probabilistic.
        confidence = float(min(100.0, max(0.0, abs(pred_ret) / 0.05 * 100.0)))

        return {
            "predicted_return": pred_ret,
            "predicted_price": pred_price,
            "current_close": close,
            "rsi": rsi,
            "confidence": confidence,
            "as_of": feat_df.index[-1],
        }
