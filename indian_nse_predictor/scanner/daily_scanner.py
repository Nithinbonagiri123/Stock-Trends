"""
Rank universe by composite score = predicted next-day log return / recent volatility.
Optionally render Plotly chart and send via Telegram; attach VaR / sector report.
"""

from __future__ import annotations

import logging
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import LOOKBACK, PARQUET_PATH, SECTOR_MAP_PATH, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, VAR_CONFIDENCE
from data.features_engineering import enrich_symbol_frame, fetch_sector_index_returns, load_sector_map
from data.store import load_panel
from risk.var_sector import risk_report_top_picks
from scanner.telegram_notify import send_photo_png
from training.pipeline import FINETUNE_W, GLOBAL_W, SCALER_FT, SCALER_G

logger = logging.getLogger(__name__)

FEATURE_COLS = ["z_ret", "log_vol", "delivery_pct", "sector_alpha", "circuit_flag"]


def _load_model_scaler():
    if FINETUNE_W.exists():
        model = tf.keras.models.load_model(FINETUNE_W)
        with SCALER_FT.open("rb") as f:
            scaler = pickle.load(f)
        return model, scaler, "finetune"
    if GLOBAL_W.exists():
        model = tf.keras.models.load_model(GLOBAL_W)
        with SCALER_G.open("rb") as f:
            scaler = pickle.load(f)
        return model, scaler, "global"
    raise FileNotFoundError("No trained model in artifacts/ — run main_train.py first.")


def _recent_vol(close: pd.Series, window: int = 20) -> float:
    r = np.log(close / close.shift(1))
    v = r.tail(window).std()
    return float(v) if pd.notna(v) else 1e-6


def scan_universe(
    panel: pd.DataFrame,
    sector_map_path: Path | None = None,
    top_n: int = 5,
) -> tuple[pd.DataFrame, dict]:
    sector_map_path = sector_map_path or SECTOR_MAP_PATH
    sm = load_sector_map(sector_map_path)
    sector_ret = fetch_sector_index_returns(sm["yahoo_sector_index"].dropna().unique().tolist())
    model, scaler, tag = _load_model_scaler()
    rows: list[dict] = []

    for sym, g in panel.groupby("symbol"):
        try:
            eg = enrich_symbol_frame(g, sm, sector_ret)
        except Exception as e:
            logger.debug("%s enrich skip: %s", sym, e)
            continue
        eg = eg.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS + ["target"])
        eg = eg[eg["circuit_flag"] < 0.5]
        if len(eg) < LOOKBACK + 2:
            continue
        X = eg[FEATURE_COLS].values.astype(np.float64)
        flat = scaler.transform(X.reshape(-1, X.shape[-1]))
        win = flat[-LOOKBACK:].reshape(1, LOOKBACK, X.shape[-1])
        pred = float(model.predict(win, verbose=0)[0, 0])
        vol = _recent_vol(eg["close"], 20)
        score = pred / (vol + 1e-8)
        rows.append(
            {
                "symbol": sym,
                "pred_log_return_1d": pred,
                "vol_20d": vol,
                "composite_score": score,
                "model": tag,
            }
        )

    if not rows:
        raise RuntimeError("No predictions — check panel and sector map coverage.")
    out = pd.DataFrame(rows).sort_values("composite_score", ascending=False).reset_index(drop=True)
    top = out.head(top_n)
    risk = risk_report_top_picks(panel, top["symbol"].tolist(), sector_map_path)
    return top, risk


def plot_top_breakouts(top: pd.DataFrame, out_path: Path) -> None:
    fig = go.Figure(
        data=[
            go.Bar(
                x=top["symbol"],
                y=top["composite_score"],
                marker_color="#2563eb",
                text=[f"{s:.3f}" for s in top["composite_score"]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Top breakout candidates (predicted return / 20d vol)",
        xaxis_title="Symbol",
        yaxis_title="Composite score",
        template="plotly_white",
        height=480,
        margin=dict(t=60, b=80),
    )
    fig.write_image(str(out_path), width=900, height=500, scale=2)


def run_daily_scan(
    panel_path: Path | None = None,
    send_telegram: bool = True,
) -> tuple[pd.DataFrame, dict]:
    panel = load_panel(panel_path or PARQUET_PATH)
    top, risk = scan_universe(panel)
    pct = int(VAR_CONFIDENCE * 100)
    caption_lines = [
        "Top 5 NSE breakout scores (pred 1d log-return / vol)",
        *[f"{i+1}. {r['symbol']}: {r['composite_score']:.4f}" for i, r in top.iterrows()],
        f"Portfolio 1d VaR ({pct}% conf. approx): {risk['var_1d_equal_weight']:.6f}",
        f"Max sector weight among picks: {risk['max_sector_weight_among_picks']:.2%}",
    ]
    caption = "\n".join(caption_lines)

    if send_telegram and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        with tempfile.TemporaryDirectory() as td:
            png = Path(td) / "top5.png"
            plot_top_breakouts(top, png)
            send_photo_png(png, caption, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    else:
        logger.info("%s", caption)

    return top, risk
