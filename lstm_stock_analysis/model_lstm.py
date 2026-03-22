"""TensorFlow / Keras LSTM architecture."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from config import DROPOUT_RATE, LEARNING_RATE, LSTM_UNITS


def build_lstm_model(
    lookback: int,
    n_features: int,
    lstm_units: tuple[int, ...] = LSTM_UNITS,
    dropout: float = DROPOUT_RATE,
) -> Sequential:
    """
    At least two LSTM layers + Dropout(0.2) + Dense(1) for next-step scaled close.
    """
    if len(lstm_units) < 2:
        raise ValueError("Use at least two LSTM layers per requirements.")

    model = Sequential()
    model.add(
        LSTM(
            lstm_units[0],
            return_sequences=True,
            input_shape=(lookback, n_features),
        )
    )
    model.add(Dropout(dropout))
    model.add(LSTM(lstm_units[1], return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def set_global_seed(seed: int) -> None:
    tf.random.set_seed(seed)
