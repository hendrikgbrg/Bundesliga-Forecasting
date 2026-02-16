import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bundesliga_forecasting.BL_config import COLUMNS, CSV_ENCODING, PATHS
from bundesliga_forecasting.BL_utils import (
    check_columns,
    ensure_dir,
    read_csv,
    save_to_csv,
)
from bundesliga_forecasting.feature_engineering.F_config import (
    MATCH_COLS,
    POST_RANK_COLS,
    PREV_RANK_COLS,
)

logger = logging.getLogger(__name__)

paths = PATHS
encoding = CSV_ENCODING
cols = COLUMNS
RANK_COLS = PREV_RANK_COLS + POST_RANK_COLS


def add_season_features(
    src_dir: Path = paths.features,
    target_dir: Path = paths.features,
    src_file: str = paths.feature_file,
    target_file: str = paths.feature_file,
) -> None:
    """
    Description:

    Usage location:
        data_editing/pipeline.py

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    logger.info("Adding season-features to the DataFrame...")
    ensure_dir([src_dir, target_dir], ["src", "target"])

    input_path = src_dir / src_file
    output_path = target_dir / target_file
    required_cols = [cols.season, cols.div, cols.date, cols.team]

    df = read_csv(input_path)
    check_columns(df, required_cols)

    calendar = _create_calendar(df)
    season_snap = _create_season_snap(df, calendar)
    season_snap = _compute_ranks(season_snap)
    season_snap = _add_table_extrema(season_snap)
    df = _merge_back(df, season_snap)
    save_to_csv(df, output_path)


#########################################################################################################


def _create_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Description:

    Usage location:
        data_edtiting/features/season.py/add_season_features

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    logger.info("Creating the season team-date calendar...")
    check_columns(df, [cols.season, cols.div, cols.date, cols.team])
    dates_df = (
        df[[cols.season, cols.div, cols.date]].drop_duplicates().reset_index(drop=True)
    )
    teams_df = (
        df[[cols.season, cols.div, cols.team]].drop_duplicates().reset_index(drop=True)
    )
    calendar = dates_df.merge(teams_df, on=[cols.season, cols.div], how="inner")

    return calendar


def _create_season_snap(df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Description:

    Usage location:
        data_edtiting/features/season.py/add_season_features

    Args:
        df (pd.DataFrame): _description_
        calendar (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    logger.info("Creating the season snap from the season team-date calendar...")
    check_columns(
        df, [cols.season, cols.div, cols.date, cols.team] + MATCH_COLS + RANK_COLS
    )
    season_snap = calendar.merge(
        df[[cols.season, cols.div, cols.date, cols.team] + MATCH_COLS + RANK_COLS],
        on=[cols.season, cols.div, cols.date, cols.team],
        how="left",
    )
    return season_snap


def _rank_by_sort_group(
    season_snap: pd.DataFrame,
    rank_cols: list[str],
    out_col: str,
    ascending: list[bool] | None = None,
) -> pd.DataFrame:
    """
    Description:

    Usage location:
        data_edtiting/features/season.py/_compute_ranks

    Args:
        season_snap (pd.DataFrame): _description_
        sort_cols (list[str]): _description_
        group_cols (list[str]): _description_
        ascending (list[bool]): _description_
        out_col (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    check_columns(
        season_snap,
        [cols.season, cols.div, cols.date, cols.team] + MATCH_COLS + RANK_COLS,
    )

    rank_group_cols = [cols.season, cols.div, cols.date]
    ffill_group_cols = [cols.season, cols.team]
    if not ascending:
        ascending_group = [True] * len(rank_group_cols)
        ascending_rank = [False] * len(rank_cols)
        ascending = ascending_group + ascending_rank
    else:
        ascending_group = ascending[: len(rank_group_cols)]
        ascending_rank = ascending[len(rank_group_cols) :]

    col_pairs = list(zip(PREV_RANK_COLS, POST_RANK_COLS))
    season_snap = season_snap.sort_values(
        by=rank_group_cols, ascending=ascending_group, kind="mergesort"
    )

    for prev_col, post_col in col_pairs:
        season_snap[post_col] = season_snap.groupby(ffill_group_cols, sort=False)[
            post_col
        ].ffill()
        season_snap[prev_col] = season_snap.groupby(ffill_group_cols, sort=False)[
            post_col
        ].shift(1)

    season_snap[MATCH_COLS + RANK_COLS] = season_snap[MATCH_COLS + RANK_COLS].fillna(0)

    season_snap = season_snap.sort_values(
        by=rank_group_cols + rank_cols, ascending=ascending, kind="mergesort"
    )

    factor = 10**3
    rev_rank_cols = rank_cols[::-1]
    season_snap["_rank_key"] = 0

    for index in range(len(rev_rank_cols)):
        season_snap["_rank_key"] += season_snap[rev_rank_cols[index]] * factor**index

    season_snap[out_col] = (
        season_snap.groupby(rank_group_cols, sort=False)["_rank_key"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    season_snap.drop(columns=["_rank_key"], inplace=True)

    season_snap = season_snap.reset_index(drop=True)

    return season_snap


def _compute_ranks(season_snap: pd.DataFrame) -> pd.DataFrame:
    """
    Description:

    Usage location:
        data_edtiting/features/season.py/add_season_features

    Args:
        season_snap (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    logger.info("Calculating ranks by sorting and grouping in the season-snap...")
    check_columns(
        season_snap,
        [cols.season, cols.div, cols.date] + PREV_RANK_COLS + POST_RANK_COLS,
    )
    # prior-match ranks
    season_snap = _rank_by_sort_group(
        season_snap,
        rank_cols=PREV_RANK_COLS,
        out_col=cols.prev_rank,
    )
    season_snap[cols.prev_trank] = np.where(
        season_snap[cols.div] == "D1",
        season_snap[cols.prev_rank],
        season_snap[cols.prev_rank] + 18,
    )

    # post-match ranks
    season_snap = _rank_by_sort_group(
        season_snap,
        rank_cols=POST_RANK_COLS,
        out_col=cols.post_rank,
    )
    season_snap[cols.post_trank] = np.where(
        season_snap[cols.div] == "D1",
        season_snap[cols.post_rank],
        season_snap[cols.post_rank] + 18,
    )
    return season_snap


def _add_table_extrema(season_snap: pd.DataFrame) -> pd.DataFrame:
    """
    Description:

    Usage location:
        data_edtiting/features/season.py/add_season_features

    Args:
        season_snap (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    logger.info("Determining total point extreme values within the season-snap...")
    check_columns(
        season_snap, [cols.div, cols.date, cols.prev_tpoints, cols.post_tpoints]
    )
    # prior-match max points
    season_snap[cols.prev_max_tpoints] = season_snap.groupby([cols.div, cols.date])[
        cols.prev_tpoints
    ].transform("max")

    # prior-match min points
    season_snap[cols.prev_min_tpoints] = season_snap.groupby([cols.div, cols.date])[
        cols.prev_tpoints
    ].transform("min")

    # post-match max points
    season_snap[cols.post_max_tpoints] = season_snap.groupby([cols.div, cols.date])[
        cols.post_tpoints
    ].transform("max")

    # post-match min points
    season_snap[cols.post_min_tpoints] = season_snap.groupby([cols.div, cols.date])[
        cols.post_tpoints
    ].transform("min")

    return season_snap


def _merge_back(df: pd.DataFrame, season_snap: pd.DataFrame) -> pd.DataFrame:
    """
    Description:

    Usage location:
        data_edtiting/features/season.py/add_season_features

    Args:
        df (pd.DataFrame): _description_
        season_snap (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    logger.info("Merging season snap back into original DataFrame...")
    merge_columns = [
        cols.season,
        cols.date,
        cols.team,
        cols.prev_min_tpoints,
        cols.prev_max_tpoints,
        cols.prev_rank,
        cols.prev_trank,
        cols.post_min_tpoints,
        cols.post_max_tpoints,
        cols.post_rank,
        cols.post_trank,
    ]
    on_columns = [cols.season, cols.date, cols.team]
    check_columns(season_snap, merge_columns)
    check_columns(df, on_columns)

    df = df.merge(
        season_snap[merge_columns],
        on=on_columns,
        how="left",
    )

    return df
