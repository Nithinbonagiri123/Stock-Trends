"""LSTM stack + multi-head attention (many-to-one regression)."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    MultiHeadAttention,
)

from config import (
    ATTENTION_HEADS,
    ATTENTION_KEY_DIM,
    DROPOUT,
    LR_INITIAL,
    LSTM_UNITS,
)


def build_model(lookback: int, n_features: int) -> Model:
    inp = Input(shape=(lookback, n_features))
    x = LSTM(LSTM_UNITS[0], return_sequences=True)(inp)
    x = Dropout(DROPOUT)(x)
    x = BatchNormalization()(x)
    x = LSTM(LSTM_UNITS[1], return_sequences=True)(x)
    x = Dropout(DROPOUT)(x)
    x = BatchNormalization()(x)

    attn = MultiHeadAttention(
        num_heads=ATTENTION_HEADS,
        key_dim=ATTENTION_KEY_DIM,
        dropout=DROPOUT,
    )(x, x)
    x = tf.keras.layers.Add()([x, attn])
    x = GlobalAveragePooling1D()(x)
    out = Dense(1, activation="linear")(x)

    model = Model(inputs=inp, outputs=out, name="mv_lstm_attention")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_INITIAL),
        loss="mse",
        metrics=["mae"],
    )
    return model
