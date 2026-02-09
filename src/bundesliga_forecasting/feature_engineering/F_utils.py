from __future__ import annotations

import logging
from typing import NamedTuple

import pandas as pd

from bundesliga_forecasting.feature_engineering.F_config import EWMA_DECAY

logger = logging.getLogger(__name__)


class OutcomeSeries(NamedTuple):
    wins: pd.Series
    draws: pd.Series
    losses: pd.Series
    games: pd.Series


def produce_outcome_series(points: pd.Series) -> OutcomeSeries:
    return OutcomeSeries(
        wins=(points == 3).astype(int),
        draws=(points == 1).astype(int),
        losses=(points == 0).astype(int),
        games=pd.Series(1, index=points.index),
    )


def grouped_aggregate(
    s: pd.Series,
    group_keys,
    *,
    window: int | None = None,
    shift: int = 0,
    min_periods: int = 1,
    clip_lower: float | None = None,
    transformer: str = "sum",
) -> pd.Series:
    g = s.groupby(group_keys, sort=False)

    if window is None:
        out = g.transform(transformer) if transformer else g.sum()
    else:
        out = (
            g.rolling(EWMA_DECAY.rolling, min_periods=min_periods)
            .sum()
            .reset_index(level=list(range(len(group_keys))), drop=True)
        )

    if shift:
        out = out.groupby(group_keys, sort=False).shift(shift, fill_value=0)

    if clip_lower is not None:
        out = out.clip(lower=clip_lower)

    return out
