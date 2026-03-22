"""Hyperparameters and symbols for the multivariate prediction engine."""

from __future__ import annotations

# Yahoo Finance symbols
SYMBOLS = ("BTC-USD", "ETH-USD", "ALI=F")
TARGET_SYMBOL = "BTC-USD"

YEARS = 10
LOOKBACK = 60

# Fractional differentiation (0 < d < 1 typical; preserves more memory than d=1)
FRAC_DIFF_D = 0.42
FRAC_DIFF_WIDTH = 50

# Fourier: number of dominant cycles to encode as sin/cos features
FOURIER_TOP_K = 5

# Walk-forward: minimum train years before first test year
MIN_TRAIN_YEARS = 5

# Model
LSTM_UNITS = (128, 64)
ATTENTION_HEADS = 4
ATTENTION_KEY_DIM = 16
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS_PER_FOLD = 80
LABEL_SMOOTHING_ALPHA = 0.08

# Learning rate schedule
LR_INITIAL = 0.001
LR_FACTOR = 0.5
LR_PATIENCE = 5
LR_MIN = 1e-6

# Callbacks
EARLY_STOP_PATIENCE = 12

RANDOM_SEED = 42

ARTIFACTS_DIR = "artifacts"
MODEL_NAME = "best_model.keras"
