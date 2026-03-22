"""
End-to-end: load data, features, train LSTM per ticker, evaluate, buy signals.

Run: python run_pipeline.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import (
    BATCH_SIZE,
    BUY_THRESHOLD,
    EPOCHS,
    LOOKBACK,
    RANDOM_SEED,
    TICKERS,
    YEARS,
)
from data_loader import download_ticker_history
from evaluation import metrics_report, plot_actual_vs_predicted, predictions_to_prices
from features import CLOSE_FEATURE_INDEX, engineer_features
from model_lstm import build_lstm_model, set_global_seed
from preprocessing import prepare_data_for_ticker
from signals import buy_signals, summarize_signals


def run_one_ticker(
    ticker: str,
    years: int,
    epochs: int,
    batch_size: int,
    out_dir: Path,
    show_plot: bool,
) -> None:
    print(f"\n=== {ticker} ===")
    raw = download_ticker_history(ticker, years=years)
    feat = engineer_features(raw)

    X_train, y_train, X_test, y_test, scaler, split_idx, close_all = prepare_data_for_ticker(
        feat
    )
    n_features = X_train.shape[2]

    set_global_seed(RANDOM_SEED)
    model = build_lstm_model(LOOKBACK, n_features)
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
    )

    pred_scaled = model.predict(X_test, verbose=0)
    y_true_p, y_pred_p = predictions_to_prices(scaler, y_test, pred_scaled)

    rep = metrics_report(y_true_p, y_pred_p)
    print(f"Test metrics: MSE={rep['mse']:.6f} RMSE={rep['rmse']:.6f} MAE={rep['mae']:.6f}")

    plot_path = out_dir / f"{ticker}_actual_vs_predicted.png"
    plot_actual_vs_predicted(
        y_true_p,
        y_pred_p,
        title=f"{ticker}: actual vs predicted close (test)",
        save_path=plot_path,
        show=show_plot,
    )
    print(f"Saved plot: {plot_path}")

    # Investment logic: compare predicted next close to *current* close at each test step
    # y_test is scaled close at time i; we need raw closes aligned with test window
    n = len(close_all)
    i0 = max(split_idx, LOOKBACK)
    test_row_indices = np.arange(i0, n)
    if len(test_row_indices) != len(y_pred_p):
        raise RuntimeError(
            f"Alignment mismatch: test steps {len(test_row_indices)} vs preds {len(y_pred_p)}"
        )
    current_close = close_all[test_row_indices - 1]
    # predicted next close for day test_row_indices is what we forecast from prior window
    sig = buy_signals(current_close, y_pred_p)
    summ = summarize_signals(sig)
    print(f"Buy signal summary (>{BUY_THRESHOLD:.0%} predicted vs current): {summ}")


def main() -> None:
    p = argparse.ArgumentParser(description="LSTM stock pipeline (yfinance + Keras)")
    p.add_argument("--tickers", nargs="+", default=TICKERS, help="Ticker symbols")
    p.add_argument("--years", type=int, default=YEARS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--no-show", action="store_true", help="Save plots only, do not open windows")
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for t in args.tickers:
        run_one_ticker(
            t.upper(),
            years=args.years,
            epochs=args.epochs,
            batch_size=args.batch_size,
            out_dir=out_dir,
            show_plot=not args.no_show,
        )


if __name__ == "__main__":
    main()
