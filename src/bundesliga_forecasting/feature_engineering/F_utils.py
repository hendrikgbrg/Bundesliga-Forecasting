from __future__ import annotations

import logging
from typing import NamedTuple

import pandas as pd

from bundesliga_forecasting.BL_utils import check_columns
from bundesliga_forecasting.feature_engineering.F_config import COLUMNS

logger = logging.getLogger(__name__)
cols = COLUMNS


class OutcomeSeries(NamedTuple):
    wins: pd.Series
    draws: pd.Series
    losses: pd.Series
    games: pd.Series
    goalsf: pd.Series
    goalsa: pd.Series


def produce_outcome_series(df: pd.DataFrame) -> OutcomeSeries:
    check_columns(df, [cols.season, cols.team, cols.points, cols.goalsf, cols.goalsa])

    return OutcomeSeries(
        wins=(df[cols.points] == 3).astype(int),
        draws=(df[cols.points] == 1).astype(int),
        losses=(df[cols.points] == 0).astype(int),
        games=pd.Series(1, index=df.index, dtype=int),
        goalsf=df[cols.goalsf],
        goalsa=df[cols.goalsa],
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
        out = g.transform(transformer)
    else:
        rolling = g.rolling(window=window, min_periods=min_periods)
        if transformer == "sum":
            out = rolling.sum()
        elif transformer == "mean":
            out = rolling.mean()
        elif transformer == "cumsum":
            out = rolling.cumsum()
        else:
            raise ValueError(f"Unsupported transformer: {transformer}")

        out = out.reset_index(level=list(range(len(group_keys))), drop=True)

    if shift:
        out = out.groupby(group_keys, sort=False).shift(shift, fill_value=0)

    if clip_lower is not None:
        out = out.clip(lower=clip_lower)

    return out
