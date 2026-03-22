"""
Walk-forward helpers: split the multi-symbol panel by calendar year so you can train/evaluate
per regime (e.g. pre- vs post-COVID). Pair with training.pipeline.train_global_nifty100 by
passing a sliced panel for each fold.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import WALK_FORWARD_TEST_YEARS


def iter_yearly_folds(
    panel: pd.DataFrame,
    test_years: int = WALK_FORWARD_TEST_YEARS,
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """
    Yield (test_year, train_panel, test_panel) where train uses all prior years and test is one year.
    """
    ix = pd.DatetimeIndex(pd.to_datetime(panel.index))
    years = sorted(ix.year.unique().tolist())
    if len(years) <= test_years:
        raise ValueError("Not enough years in panel for walk-forward.")
    for i in range(len(years) - test_years):
        train_years = years[: i + test_years]
        test_y = years[i + test_years]
        tr = panel.loc[ix.year.isin(train_years)]
        te = panel.loc[ix.year == test_y]
        yield int(test_y), tr, te
