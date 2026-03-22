# Quantitative stock pipeline (LSTM + attention)

Production-style **Python** layout for: market-context data, **pandas_ta** features, **TensorFlow** LSTM with **multi-head attention**, **EarlyStopping**, and a **rule-based backtest** (predicted gain + RSI gate) with **Sharpe** and **max drawdown**.

**Not financial advice.** Past performance does not guarantee future results.

## Modules

| Class | File | Role |
|--------|------|------|
| `DataIngestion` | `data_ingestion.py` | `yfinance` download: target + `^GSPC`, `^VIX`, `^TNX`; align to equity calendar and **forward-fill** context |
| `FeatureEngineering` | `feature_engineering.py` | RSI(14), MACD, Bollinger(20,2), ATR(14); Close lags (1,2,3,5); Volume ROC; **Target** = next-day return |
| `ModelTrainer` | `model_trainer.py` | `MinMaxScaler` (fit on train rows), sequences, **two LSTM + Dropout + BatchNorm + MultiHeadAttention + Dense**, `EarlyStopping` on `val_loss` |
| `Backtester` | `backtester.py` | **BUY** only if predicted return > **1.5%** and **RSI < 70**; Sharpe (annualized), cumulative return, max drawdown |

## Run

```bash
cd quant_system
pip install -r requirements.txt
python main.py --ticker AAPL
```

Options: `--epochs 120` (EarlyStopping may stop earlier).

## Automated daily pipeline (`run_daily.py`)

Runs **without manual input**: fetches **intraday** for today (merged into 10y history), **loads** `automation/artifacts/model.keras` + `scaler.pkl` if present or **trains** once, predicts **T+1** return, **logs** to SQLite (`automation/prediction_log.sqlite`), optional **Telegram / email** on **BUY** (predicted gain **> 1.5%** and **RSI < 70**). **Retries** failed runs after **5 minutes** (configurable).

```bash
cd quant_system
cp .env.example .env   # set PIPELINE_TICKER, optional Telegram/email
pip install -r requirements.txt
python run_daily.py --once    # test one run
python run_daily.py           # APScheduler: weekdays at SCHEDULE_HOUR (default 15:00 NY)
```

Artifacts and DB live under `quant_system/automation/`. See `.env.example` for all variables.

## Design notes

- **Scaler**: Fitted only on the **first TRAIN_RATIO rows** of the feature matrix (before sequences), then all rows are transformed — limits leakage versus scaling on the full sample.
- **Sequences**: Causal windows `features[i-lookback:i]` → predict `Target[i]` (next-day return).
- **Backtest**: Applied on the **test** split only; strategy return is the **realized** next-day return when a signal fires, else **0** (cash).
