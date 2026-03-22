"""
Single job: live data → features → train or load → predict → log → optional alerts.
"""

from __future__ import annotations

import logging
import math
import os
import time

from config import EPOCHS

from .config_auto import (
    ARTIFACTS_DIR,
    BUY_THRESHOLD,
    DB_PATH,
    MAX_RETRIES,
    MAX_RSI_ALERT,
    PIPELINE_TICKER,
    RETRY_DELAY_SECONDS,
)
from .live_data import build_live_master
from .logging_db import PredictionLogger
from .notifications import alert_buy_signal
from feature_engineering import FeatureEngineering, FeatureEngineeringError
from model_trainer import ModelTrainer, ModelTrainerError

logger = logging.getLogger(__name__)


def _artifacts_ready() -> bool:
    return (
        (ARTIFACTS_DIR / "model.keras").exists()
        and (ARTIFACTS_DIR / "scaler.pkl").exists()
        and (ARTIFACTS_DIR / "feature_columns.json").exists()
    )


def run_once() -> None:
    ticker = PIPELINE_TICKER
    if not ticker:
        raise ValueError("PIPELINE_TICKER is empty.")

    logger.info("Pipeline run for %s", ticker)

    master = build_live_master(ticker, years=int(os.environ.get("YEARS_HISTORY", "10")))
    fe = FeatureEngineering()
    feat = fe.transform(master)
    cols = fe.feature_columns_for_model(feat)

    plog = PredictionLogger(DB_PATH)
    plog.update_pending_actuals(feat, ticker)

    epochs = int(os.environ.get("EPOCHS", EPOCHS))
    trainer = ModelTrainer(epochs=epochs)

    if _artifacts_ready():
        logger.info("Loading saved model from %s", ARTIFACTS_DIR)
        trainer.load_artifacts(ARTIFACTS_DIR)
    else:
        logger.info("No saved artifacts; training on 10y history (may take a while)…")
        trainer.train_from_scratch(feat, cols, ARTIFACTS_DIR)

    out = trainer.predict_next(feat)

    rsi = out["rsi"]
    rsi_ok = (not math.isnan(rsi)) and (rsi < MAX_RSI_ALERT)
    buy = (out["predicted_return"] > BUY_THRESHOLD) and rsi_ok

    plog.insert_run(
        ticker=ticker,
        as_of=out["as_of"],
        predicted_return=out["predicted_return"],
        predicted_price=out["predicted_price"],
        current_close=out["current_close"],
        rsi=rsi,
        confidence=out["confidence"],
        buy_signal=buy,
    )

    if buy:
        logger.info("BUY signal — sending notifications.")
        alert_buy_signal(
            ticker,
            out["predicted_price"],
            rsi,
            out["predicted_return"],
            out["confidence"],
        )


def run_pipeline_with_retry() -> None:
    """Run the job with retries (e.g. network blips)."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            run_once()
            return
        except (FeatureEngineeringError, ModelTrainerError, ValueError, OSError) as e:
            last_exc = e
            logger.exception("Attempt %s/%s failed: %s", attempt, MAX_RETRIES, e)
        except Exception as e:
            last_exc = e
            logger.exception("Attempt %s/%s failed (unexpected): %s", attempt, MAX_RETRIES, e)

        if attempt < MAX_RETRIES:
            logger.info("Retrying in %s seconds…", RETRY_DELAY_SECONDS)
            time.sleep(RETRY_DELAY_SECONDS)

    if last_exc is not None:
        try:
            PredictionLogger(DB_PATH).log_exception(PIPELINE_TICKER, last_exc)
        except Exception as log_e:
            logger.error("Could not log failure to DB: %s", log_e)
        raise last_exc
