"""
SQLite log for predictions, actuals (filled next run), and confidence.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PredictionLogger:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_ts TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    predicted_return REAL,
                    predicted_price REAL,
                    current_close REAL,
                    rsi REAL,
                    confidence REAL,
                    buy_signal INTEGER,
                    actual_return REAL,
                    actual_filled_ts TEXT,
                    error TEXT,
                    extra_json TEXT
                )
                """
            )
            conn.commit()

    def insert_run(
        self,
        ticker: str,
        as_of: Any,
        predicted_return: float,
        predicted_price: float,
        current_close: float,
        rsi: float,
        confidence: float,
        buy_signal: bool,
        error: str | None = None,
        extra: dict | None = None,
    ) -> int:
        run_ts = _utc_now_iso()
        as_of_s = str(pd.Timestamp(as_of).date())
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO prediction_log (
                    run_ts, ticker, as_of, predicted_return, predicted_price,
                    current_close, rsi, confidence, buy_signal, actual_return,
                    actual_filled_ts, error, extra_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_ts,
                    ticker,
                    as_of_s,
                    predicted_return,
                    predicted_price,
                    current_close,
                    rsi,
                    confidence,
                    int(bool(buy_signal)),
                    None,
                    None,
                    error,
                    json.dumps(extra) if extra else None,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def update_pending_actuals(self, feat_df: pd.DataFrame, ticker: str) -> int:
        """
        For rows with NULL actual_return, fill using next trading day's Adj Close
        when available in feat_df (sorted index).
        """
        if feat_df.empty or "Adj Close" not in feat_df.columns:
            return 0
        df = feat_df.sort_index()
        idx_list = list(df.index)
        ac = df["Adj Close"].astype(float)

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, as_of FROM prediction_log
                WHERE ticker = ? AND actual_return IS NULL AND error IS NULL
                ORDER BY id ASC
                """,
                (ticker,),
            ).fetchall()

        updated = 0
        for row_id, as_of_s in rows:
            try:
                target_d = pd.Timestamp(as_of_s).date()
            except Exception:
                continue
            pos = None
            for i, ts in enumerate(idx_list):
                if pd.Timestamp(ts).date() == target_d:
                    pos = i
                    break
            if pos is None:
                continue
            if pos + 1 >= len(idx_list):
                continue
            c0 = float(ac.loc[idx_list[pos]])
            c1 = float(ac.loc[idx_list[pos + 1]])
            actual = c1 / c0 - 1.0
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE prediction_log
                    SET actual_return = ?, actual_filled_ts = ?
                    WHERE id = ?
                    """,
                    (actual, _utc_now_iso(), row_id),
                )
                conn.commit()
            updated += 1
            logger.info("Filled actual_return=%.6f for log id=%s", actual, row_id)

        return updated

    def log_exception(self, ticker: str, exc: Exception) -> None:
        """Record a failed run (e.g. after retries exhausted)."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO prediction_log (run_ts, ticker, as_of, error)
                VALUES (?,?,?,?)
                """,
                (_utc_now_iso(), ticker, "ERROR", repr(exc)[:2000]),
            )
            conn.commit()
