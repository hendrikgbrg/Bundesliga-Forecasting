import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bundesliga_forecasting.feature_engineering.feature_config import (
    COLUMNS,
    MATCH_COLS,
    POST_RANK_COLS,
    PRIOR_RANK_COLS,
)
from bundesliga_forecasting.project_config import CSV_ENCODING, PATHS
from bundesliga_forecasting.project_utils import (
    check_columns,
    ensure_dir,
    read_csv,
    save_to_csv,
)

logger = logging.getLogger(__name__)

paths = PATHS
encoding = CSV_ENCODING
cols = COLUMNS
RANK_COLS = PRIOR_RANK_COLS + POST_RANK_COLS


def add_season_features(
    src_dir: Path = paths.features,
    target_dir: Path = paths.features,
    src_file: str = paths.f_filename,
    target_file: str = paths.f_filename,
) -> pd.DataFrame:
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
    return df


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
    dates_df = df[[cols.season, cols.div, cols.date]].drop_duplicates()
    teams_df = df[[cols.season, cols.div, cols.team]].drop_duplicates()
    calendar = dates_df.merge(teams_df, on=[cols.div, cols.season], how="inner")

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
    logger.info("Creating season snap...")
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
    sort_cols: list[str],
    group_cols: list[str],
    ascending: list[bool],
    out_col: str,
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
    logger.info("Applying rank function...")
    season_snap = season_snap.sort_values(
        by=sort_cols, ascending=ascending, kind="mergesort"
    )

    season_snap[RANK_COLS] = season_snap.groupby([cols.season, cols.team])[
        RANK_COLS
    ].ffill()

    season_snap[MATCH_COLS + RANK_COLS].fillna(0, inplace=True)

    season_snap[out_col] = season_snap.groupby(group_cols, sort=False).ngroup().add(1)

    season_snap.reset_index(drop=True, inplace=True)

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
    logger.info("Computing ranks by sorting and grouping...")

    # prior-match ranks
    season_snap = _rank_by_sort_group(
        season_snap,
        sort_cols=[cols.season, cols.div, cols.date] + PRIOR_RANK_COLS,
        group_cols=[cols.season, cols.div, cols.date],
        ascending=[True, True, True, False, False, False],
        out_col=cols.pre_rank,
    )
    season_snap[cols.pre_trank] = np.where(
        season_snap[cols.div] == "D1",
        season_snap[cols.pre_rank],
        season_snap[cols.pre_rank] + 18,
    )

    # post-match ranks
    season_snap = _rank_by_sort_group(
        season_snap,
        sort_cols=[cols.season, cols.div, cols.date] + POST_RANK_COLS,
        group_cols=[cols.season, cols.div, cols.date],
        ascending=[True, True, True, False, False, False],
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
    logger.info("Adding total point extreme values to the table...")
    # prior-match max points
    season_snap[cols.pre_max_tpoints] = season_snap.groupby([cols.div, cols.date])[
        cols.pre_tpoints
    ].transform("max")

    # prior-match min points
    season_snap[cols.pre_min_tpoints] = season_snap.groupby([cols.div, cols.date])[
        cols.pre_tpoints
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
    logger.info("Merging season snap back into original DataFrane...")
    df = df.merge(
        season_snap[
            [
                cols.season,
                cols.date,
                cols.team,
                cols.pre_min_tpoints,
                cols.pre_max_tpoints,
                cols.pre_rank,
                cols.pre_trank,
                cols.post_min_tpoints,
                cols.post_max_tpoints,
                cols.post_rank,
                cols.post_trank,
            ]
        ],
        on=[cols.season, cols.date, cols.team],
        how="left",
    )

    return df
