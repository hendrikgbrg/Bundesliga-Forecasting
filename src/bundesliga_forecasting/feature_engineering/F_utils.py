from __future__ import annotations

import logging
from typing import NamedTuple

import pandas as pd

from bundesliga_forecasting.BL_utils import check_columns
from bundesliga_forecasting.feature_engineering.F_config import (
    COLUMNS,
    POST_RANK_COLS,
    PREV_RANK_COLS,
)

logger = logging.getLogger(__name__)
cols = COLUMNS
RANK_COLS = PREV_RANK_COLS + POST_RANK_COLS


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


def create_season_end(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:

    ## Internal function ##
    def _create_calendar(df: pd.DataFrame) -> pd.DataFrame:
        seasons = df[cols.season].drop_duplicates()
        teams = df[cols.team].drop_duplicates()
        calendar = seasons.to_frame().merge(teams.to_frame(), how="cross", sort=False)
        return calendar

    ## Main function ##
    check_columns(df, required_cols)
    calendar = _create_calendar(df)
    season_end = (
        df[required_cols]
        .groupby([cols.season, cols.team], as_index=False, sort=False)
        .last()
    )
    season_end = calendar.merge(
        season_end, on=[cols.season, cols.team], how="left", sort=False
    ).fillna(0)

    return season_end


def prev_season_value(df: pd.DataFrame, new_col: str, ref_col: str) -> pd.DataFrame:
    df = df.copy()
    df["prev_season"] = df[cols.season] - 1

    prev_lookup = df[[cols.team, cols.season, ref_col]].rename(
        columns={cols.season: "prev_season", ref_col: new_col}
    )
    df = df.merge(prev_lookup, on=[cols.team, "prev_season"], how="left")
    return df


def merge_back(
    df: pd.DataFrame,
    season_end: pd.DataFrame,
    *,
    merge_cols: list[str],
    merge_on: list[str],
) -> pd.DataFrame:
    check_columns(season_end, merge_cols)
    check_columns(df, merge_on)

    df = df.merge(season_end[merge_cols], on=merge_on, how="left")
    return df
