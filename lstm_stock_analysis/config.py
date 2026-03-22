"""Shared configuration for the LSTM stock pipeline."""

# Default tickers (10 years of daily data each)
TICKERS = ["AAPL", "MSFT", "GOOGL"]

# History length in years
YEARS = 10

# LSTM sequence length (trading days)
LOOKBACK = 60

# Train / test split ratio (chronological)
TRAIN_RATIO = 0.8

# Model
LSTM_UNITS = (64, 32)
DROPOUT_RATE = 0.2
EPOCHS = 80
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Buy signal: predict next close > current close * (1 + threshold)
BUY_THRESHOLD = 0.02

# Reproducibility
RANDOM_SEED = 42
