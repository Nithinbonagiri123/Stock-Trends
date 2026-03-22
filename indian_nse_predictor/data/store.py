"""
Append / read Parquet store for multi-symbol OHLCV.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _dedupe_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-symbol panels share the same calendar index; never dedupe on index alone
    (that would keep one stock per day and wipe the rest).
    """
    if "symbol" not in df.columns:
        return df[~df.index.duplicated(keep="last")]
    tmp = df.reset_index()
    date_col = tmp.columns[0]
    tmp = tmp.drop_duplicates(subset=[date_col, "symbol"], keep="last")
    return tmp.set_index(date_col).sort_index()


def frames_to_parquet(frames: list[pd.DataFrame], path: Path) -> None:
    if not frames:
        logger.warning("No frames to write.")
        return
    df = pd.concat(frames, axis=0, ignore_index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df], axis=0)
    df = _dedupe_panel(df)
    df.to_parquet(path, compression="snappy", index=True)
    logger.info("Wrote %s rows → %s", len(df), path)


def append_to_hdf5(frames: list[pd.DataFrame], path: Path) -> None:
    """Optional compressed HDF5 store (key `daily`) for fast I/O."""
    if not frames:
        return
    df = pd.concat(frames, axis=0, ignore_index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_hdf(path, key="daily")
        df = pd.concat([old, df], axis=0)
    df = _dedupe_panel(df)
    df.to_hdf(path, key="daily", mode="w", complevel=5, complib="zlib")
    logger.info("HDF5 %s rows → %s", len(df), path)


def load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    s = str(path).lower()
    if s.endswith(".h5") or s.endswith(".hdf5"):
        return pd.read_hdf(path, key="daily")
    return pd.read_parquet(path)
