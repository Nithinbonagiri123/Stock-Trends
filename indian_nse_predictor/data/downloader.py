"""
Parallel yfinance downloader with tqdm progress.
"""

from __future__ import annotations

import logging
import time
from functools import partial
from multiprocessing import Pool, cpu_count
import pandas as pd
import yfinance as yf
from tqdm import tqdm

logger = logging.getLogger(__name__)

PERIOD = "10y"
INTERVAL = "1d"


def _fetch_one(symbol: str, pause: float = 0.05) -> pd.DataFrame | None:
    time.sleep(pause)
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=PERIOD, interval=INTERVAL, auto_adjust=True)
        if df is None or df.empty:
            return None
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df["symbol"] = symbol
        df = df.rename(columns=str.lower)
        return df
    except Exception as e:
        logger.debug("%s: %s", symbol, e)
        return None


def download_parallel(
    symbols: list[str],
    workers: int | None = None,
) -> list[pd.DataFrame]:
    workers = workers or max(1, cpu_count() - 1)
    fn = partial(_fetch_one)
    chunks: list[pd.DataFrame] = []

    with Pool(workers) as pool:
        for df in tqdm(
            pool.imap_unordered(fn, symbols, chunksize=2),
            total=len(symbols),
            desc="yfinance",
            unit="sym",
        ):
            if df is not None and not df.empty:
                chunks.append(df)
    return chunks
