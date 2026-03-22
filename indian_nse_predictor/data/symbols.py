"""
Fetch NSE equity symbols via nsetools (fallback: nsepython / demo list).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import SYMBOLS_CACHE, YF_SUFFIX

logger = logging.getLogger(__name__)


def _with_suffix(codes: list[str]) -> list[str]:
    out = []
    for c in codes:
        c = str(c).strip().upper()
        if not c or c == "SYMBOL":
            continue
        if not c.endswith(YF_SUFFIX):
            c = f"{c}{YF_SUFFIX}"
        out.append(c)
    return sorted(set(out))


def fetch_nse_symbol_list() -> list[str]:
    """Return Yahoo-format symbols e.g. RELIANCE.NS"""
    try:
        from nsetools import Nse

        nse = Nse()
        codes = nse.get_stock_codes()
        raw = list(codes.keys()) if isinstance(codes, dict) else list(codes)
        logger.info("nsetools: fetched %s raw symbols", len(raw))
    except Exception as e:
        logger.warning("nsetools failed (%s); trying fallbacks.", e)
        raw = []

    if not raw:
        try:
            import nsepython as nsep  # type: ignore

            if hasattr(nsep, "nse_eq_symbols"):
                raw = list(nsep.nse_eq_symbols())  # type: ignore
            elif hasattr(nsep, "equity_list"):
                raw = list(nsep.equity_list())  # type: ignore
            logger.info("nsepython: fetched %s symbols", len(raw))
        except Exception as e2:
            logger.warning("nsepython not available (%s)", e2)
            raw = []

    if not raw:
        logger.warning("Using demo symbol list — install nsetools and check connectivity.")
        raw = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC"]

    return _with_suffix(raw)


def load_or_fetch_symbols(cache_path: Path = SYMBOLS_CACHE) -> list[str]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            data = json.load(f)
        return list(data["symbols"])
    syms = fetch_nse_symbol_list()
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump({"symbols": syms, "count": len(syms)}, f, indent=2)
    return syms
