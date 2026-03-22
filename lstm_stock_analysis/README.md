# LSTM stock analysis (Python)

Educational pipeline: download daily prices with **yfinance**, add **RSI / SMA / MACD**, train a **stacked LSTM** on scaled features, plot **actual vs predicted** prices, and print **rule-based buy signals** (predicted next close more than 2% above current close).

**Not financial advice.** Past-model fit does not guarantee future performance.

## Setup

The Python code **already lives in your project** under `stock-ai-trends/lstm_stock_analysis/`. Nothing has to run “outside” that folder.

A **virtual environment** was suggested only to keep packages (TensorFlow, etc.) **separate from your system Python**, so versions don’t clash with other projects. It is **optional**.

### Option A — Install into your normal Python (no venv)

From the `lstm_stock_analysis` folder:

```bash
cd lstm_stock_analysis
pip install -r requirements.txt
```

Then run with the same `python` you used for `pip` (e.g. `python3 run_pipeline.py`).

### Option B — Virtual env *inside* the project (still “in your project file”)

If you use a venv, it is usually a folder like `lstm_stock_analysis/.venv` — that **is** inside your project; it only holds dependencies so they don’t pollute the rest of your machine.

```bash
cd lstm_stock_analysis
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Always run from the `lstm_stock_analysis` directory (where `run_pipeline.py` is):

```bash
cd lstm_stock_analysis
python run_pipeline.py
```

Options:

```bash
python run_pipeline.py --tickers AAPL MSFT GOOGL --epochs 50 --out-dir outputs --no-show
```

- Plots are saved under `outputs/<TICKER>_actual_vs_predicted.png`.
- `--no-show` only saves figures (useful on servers).

## Layout

| Module | Role |
|--------|------|
| `config.py` | Tickers, lookback, split ratio, LSTM hyperparameters, buy threshold |
| `data_loader.py` | `yfinance` download (~10 years daily) |
| `features.py` | RSI(14), SMA(50/200), MACD(12,26,9) |
| `preprocessing.py` | `MinMaxScaler` fit on train only, 80/20 chronological split, sequences |
| `model_lstm.py` | Two LSTM layers + `Dropout(0.2)` + `Dense(1)` |
| `evaluation.py` | Metrics + Matplotlib actual vs predicted |
| `signals.py` | Buy if predicted next price exceeds 1.02 × current close |
| `run_pipeline.py` | CLI entrypoint |

## Notes

- The scaler is **fit only on training rows** to reduce leakage; sequences in the test set can use history from before the split.
- Training uses a 10% validation split from the training tensor for early monitoring only (not the final test period).
