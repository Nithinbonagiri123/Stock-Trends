# Multivariate quant engine (BTC / ETH / Aluminium)

Educational **TensorFlow** pipeline implementing:

- **Fractional differentiation** (FFD) on log prices for stationarity with memory
- **Log returns** per asset (volatility-normalized vs raw levels)
- **Fourier** sin/cos features from dominant FFT cycles of BTC log price
- **pandas_ta** RSI (+ volume ROC where available)
- **LSTM + multi-head attention**, **many-to-one** predicting **next-day BTC-USD log return** using stacked features from **BTC-USD, ETH-USD, ALI=F**
- **Walk-forward** folds (train years 1…k, test year k+1; minimum train span configurable)
- **Label smoothing** on training targets
- **ReduceLROnPlateau** + **EarlyStopping**
- **Sharpe-based** `ModelCheckpoint` analog: saves `artifacts/best_model.keras` when **validation Sharpe** (sign(pred)×y) improves

**Not financial advice.** Research-grade scaffolding; validate before any real use.

## Run

```bash
cd quant_multivariate
python3 -m pip install -r requirements.txt
python3 main.py
```

## Symbols

- `BTC-USD`, `ETH-USD`, `ALI=F` (Yahoo). If a symbol fails, check Yahoo Finance ticker names.

## Outputs

- Console JSON with per-fold **test Sharpe** and **MSE**
- `artifacts/best_model.keras` — last improved Sharpe checkpoint across training
