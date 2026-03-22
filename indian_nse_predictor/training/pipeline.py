"""
Build sequences, walk-forward (1y test), global train on Nifty100, optional fine-tune.
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    BATCH_SIZE,
    EPOCHS_FINETUNE,
    EPOCHS_GLOBAL,
    LOOKBACK,
    SECTOR_MAP_PATH,
    WALK_FORWARD_TEST_YEARS,
)
from data.features_engineering import (
    enrich_symbol_frame,
    fetch_sector_index_returns,
    load_sector_map,
)
from models.lstm_transformer import build_lstm_transformer

logger = logging.getLogger(__name__)

ART = _ROOT / "artifacts"
GLOBAL_W = ART / "global_model.keras"
FINETUNE_W = ART / "finetune_model.keras"
SCALER_G = ART / "scaler_global.pkl"
SCALER_FT = ART / "scaler_finetune.pkl"


def _sequences(X: np.ndarray, y: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(lookback, len(X)):
        xs.append(X[i - lookback : i])
        ys.append(y[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _panel_to_xy(
    panel: pd.DataFrame,
    sector_map,
    sector_ret: dict,
    nifty_set: set[str] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    feats_all: list[np.ndarray] = []
    ys_all: list[np.ndarray] = []
    feature_cols = ["z_ret", "log_vol", "delivery_pct", "sector_alpha", "circuit_flag"]

    for sym, g in panel.groupby("symbol"):
        if nifty_set is not None and sym not in nifty_set:
            continue
        try:
            eg = enrich_symbol_frame(g, sector_map, sector_ret)
        except Exception as e:
            logger.debug("skip %s: %s", sym, e)
            continue
        eg = eg.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols + ["target"])
        eg = eg[eg["circuit_flag"] < 0.5]
        if len(eg) < LOOKBACK + 5:
            continue
        X = eg[feature_cols].values.astype(np.float64)
        y = eg["target"].values.astype(np.float64)
        xs, ys = _sequences(X, y, LOOKBACK)
        if len(xs) < 10:
            continue
        feats_all.append(xs)
        ys_all.append(ys)
    if not feats_all:
        return None
    X = np.concatenate(feats_all, axis=0)
    y = np.concatenate(ys_all, axis=0)
    return X, y


def walk_forward_split(
    dates: pd.DatetimeIndex,
    test_years: int = WALK_FORWARD_TEST_YEARS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, test_idx) boolean masks on *sequence* rows (caller maps)."""
    years = sorted(pd.unique(dates.year))
    folds = []
    for i in range(len(years) - test_years):
        train_years = years[: i + test_years]
        test_y = years[i + test_years]
        tr = dates.year.isin(train_years)
        te = dates.year == test_y
        folds.append((tr.values, te.values))
    return folds


def train_global_nifty100(
    panel: pd.DataFrame,
    nifty_symbols: list[str],
    sector_map_path: Path | None = None,
) -> None:
    sector_map_path = sector_map_path or SECTOR_MAP_PATH
    ART.mkdir(parents=True, exist_ok=True)
    sm = load_sector_map(sector_map_path)
    ix_list = sm["yahoo_sector_index"].dropna().unique().tolist()
    sector_ret = fetch_sector_index_returns(ix_list)
    nifty_set = {f"{s.strip().upper()}.NS" for s in nifty_symbols}
    in_panel = nifty_set & set(panel["symbol"].unique())
    logger.info("Nifty symbols in panel: %s / %s", len(in_panel), len(nifty_set))
    if not in_panel:
        raise RuntimeError(
            "No overlap between Nifty list and downloaded panel — re-run main_download.py "
            "(it prioritizes Nifty names) or raise MAX_SYMBOLS."
        )
    if len(in_panel) < 5:
        raise RuntimeError(
            f"Only {len(in_panel)} Nifty names in panel (expected many). Your Parquet may be "
            "corrupt: an older bug deduped by date only and kept one symbol per day. "
            "Delete data_store/nse_daily.parquet and run main_download.py again."
        )

    xy = _panel_to_xy(panel, sm, sector_ret, nifty_set)
    if xy is None:
        raise RuntimeError(
            "No training samples for Nifty100 subset — check data & sector map, or re-download "
            "after fixing Parquet (see error above if Nifty count in panel was low)."
        )
    X, y = xy
    scaler = StandardScaler()
    X2 = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.fit_transform(X2).reshape(X.shape)

    split = int(len(X_scaled) * 0.9)
    X_train, X_val = X_scaled[:split], X_scaled[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_lstm_transformer(LOOKBACK, X.shape[2])
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_GLOBAL,
        batch_size=BATCH_SIZE,
        verbose=1,
    )
    model.save(GLOBAL_W)
    with SCALER_G.open("wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved global model → %s scaler → %s", GLOBAL_W, SCALER_G)


def finetune_on_universe(panel: pd.DataFrame, sector_map_path: Path | None = None) -> None:
    sector_map_path = sector_map_path or SECTOR_MAP_PATH
    if not GLOBAL_W.exists():
        raise FileNotFoundError("Train global model first.")
    sm = load_sector_map(sector_map_path)
    sector_ret = fetch_sector_index_returns(sm["yahoo_sector_index"].dropna().unique().tolist())
    xy = _panel_to_xy(panel, sm, sector_ret, nifty_set=None)
    if xy is None:
        raise RuntimeError("No samples for fine-tune.")
    X, y = xy
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    model = tf.keras.models.load_model(GLOBAL_W)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse", metrics=["mae"])
    split = int(len(Xs) * 0.9)
    model.fit(
        Xs[:split],
        y[:split],
        validation_data=(Xs[split:], y[split:]),
        epochs=EPOCHS_FINETUNE,
        batch_size=BATCH_SIZE,
        verbose=1,
    )
    model.save(FINETUNE_W)
    with SCALER_FT.open("wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved fine-tuned model → %s scaler → %s", FINETUNE_W, SCALER_FT)
