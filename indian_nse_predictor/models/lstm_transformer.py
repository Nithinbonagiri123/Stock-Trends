"""Global LSTM + multi-head attention (Transformer-style) regressor."""

from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    MultiHeadAttention,
)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DROPOUT, LSTM_UNITS, TRANSFORMER_FF, TRANSFORMER_HEADS


def build_lstm_transformer(lookback: int, n_features: int) -> Model:
    inp = Input(shape=(lookback, n_features))
    x = LSTM(LSTM_UNITS, return_sequences=True)(inp)
    x = Dropout(DROPOUT)(x)
    attn = MultiHeadAttention(
        num_heads=TRANSFORMER_HEADS,
        key_dim=TRANSFORMER_FF // TRANSFORMER_HEADS,
        dropout=DROPOUT,
    )(x, x)
    x = tf.keras.layers.Add()([x, attn])
    x = tf.keras.layers.LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    out = Dense(1, activation="linear")(x)
    model = Model(inputs=inp, outputs=out, name="lstm_transformer_global")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model
