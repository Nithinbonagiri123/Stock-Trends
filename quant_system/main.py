"""
Entry point: ingest data, engineer features, train LSTM+Attention, run backtest.

Usage:
    python main.py --ticker AAPL
    python main.py --ticker MSFT --epochs 60
"""

from __future__ import annotations

import argparse
import logging
import sys

from config import DEFAULT_TICKER, EPOCHS
from data_ingestion import DataIngestion, DataIngestionError
from feature_engineering import FeatureEngineering, FeatureEngineeringError
from model_trainer import ModelTrainer, ModelTrainerError
from backtester import Backtester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("quant_system")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Production-style quant pipeline (LSTM + attention)")
    p.add_argument("--ticker", type=str, default=DEFAULT_TICKER, help="Target symbol (e.g. AAPL)")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs (EarlyStopping may stop sooner)")
    args = p.parse_args(argv)

    ticker = args.ticker.strip().upper()

    try:
        logger.info("Downloading market data for %s + context indices…", ticker)
        ingest = DataIngestion(ticker=ticker)
        raw = ingest.build_master_frame()

        logger.info("Engineering features (pandas_ta)…")
        fe = FeatureEngineering()
        feat_df = fe.transform(raw)
        feature_cols = fe.feature_columns_for_model(feat_df)
        logger.info("Using %d numeric features.", len(feature_cols))

        logger.info("Training model (LSTM + attention, EarlyStopping on val_loss)…")
        trainer = ModelTrainer(epochs=args.epochs)
        X, y, rsi_seq, _ = trainer.prepare_arrays(feat_df, feature_cols)
        out = trainer.train_and_evaluate(X, y, rsi_seq)

        logger.info("Running backtest on held-out test period…")
        bt = Backtester()
        result, _ = bt.run(
            out["pred_test"],
            out["y_test"],
            out["rsi_test"],
        )

        print("\n--- Test-set backtest (illustrative, not financial advice) ---")
        print(f"Steps:           {result.n_steps}")
        print(f"Buy signals:     {result.n_signals}")
        print(f"Sharpe (ann.):   {result.sharpe_ratio:.4f}")
        print(f"Max drawdown:    {result.max_drawdown:.4f}")
        print(f"Cumulative ret:  {result.cumulative_return:.4f}")
        print(f"Hit rate (|sig): {result.hit_rate:.4f}")
        print("---------------------------------------------------------------\n")

    except (
        DataIngestionError,
        FeatureEngineeringError,
        ModelTrainerError,
        ValueError,
    ) as e:
        logger.error("%s", e)
        return 1
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
