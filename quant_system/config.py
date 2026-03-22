"""Global configuration for the quantitative pipeline."""

from __future__ import annotations

# Target equity (e.g. single stock or ETF)
DEFAULT_TICKER = "AAPL"

# Market context (Yahoo Finance symbols)
CONTEXT_SYMBOLS = ("^GSPC", "^VIX", "^TNX")

YEARS_HISTORY = 10
INTERVAL = "1d"

# Sequence length for LSTM
LOOKBACK = 60

# Train / validation / test split (chronological fractions of cleaned rows)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15  # remainder is test

# Model
LSTM_UNITS = (64, 32)
DROPOUT_RATE = 0.2
ATTENTION_HEADS = 4
ATTENTION_KEY_DIM = 8
BATCH_SIZE = 32
EPOCHS = 120
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15

# Backtest rules
MIN_PREDICTED_GAIN = 0.015  # 1.5%
MAX_RSI_FOR_BUY = 70.0

RANDOM_SEED = 42
