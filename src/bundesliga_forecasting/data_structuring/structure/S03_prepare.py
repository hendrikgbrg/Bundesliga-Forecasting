from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from bundesliga_forecasting.BL_config import DATE_COL, PATHS
from bundesliga_forecasting.BL_utils import (
    df_sort,
    ensure_dir,
    read_csv,
    save_to_csv,
)
from bundesliga_forecasting.data_structuring.S_config import (
    COLUMNS,
    SEASON_COL,
    SEASON_START_MONTH,
)

logger = logging.getLogger(__name__)

paths = PATHS
cols = COLUMNS


def prepare(
    src_dir: Path = paths.merged,
    target_dir: Path = paths.prepared,
    src_file: str = paths.m_filename,
    target_file: str = paths.p_filename,
) -> None:
    """
    Description:

    Usage location:
        data_creation/pipeline.py

    Args:
        df (pd.DataFrame): _description_
        add_season (_type_): _description_
        team_match_split (_type_): _description_
    """
    logger.info("Starting file priorparation...")

    ensure_dir([src_dir, target_dir], ["src", "target"])

    input_path = src_dir / src_file
    output_path = target_dir / target_file

    df = read_csv(input_path)
    df = _add_season(df)
    df = _team_match_split(df)
    df = df_sort(df, sort_cols=cols.sort_by)

    save_to_csv(df, output_path)

    logger.info("File prepared successfully and saved to %s", output_path)


##############################################################################


def _add_season(
    df: pd.DataFrame,
    *,
    date_col: str = DATE_COL,
    season_col: str = SEASON_COL,
    season_start: int = SEASON_START_MONTH,
) -> pd.DataFrame:
    """
    Description:

    Usage location:
        data_creation/io/prepare.py

    Args:
        df (pd.DataFrame): _description_
        date_col (str, optional): _description_. Defaults to "Date".
        season_col (str, optional): _description_. Defaults to "Season".
        eason_start (int, optional): _description_. Defaults to 7.

    Returns:
        pd.DataFrame: _description_
    """

    if date_col not in df.columns:
        raise KeyError(
            f"The column '{date_col} could not be found in the given DataFrame."
        )

    if not is_datetime64_any_dtype(df[date_col]):
        raise ValueError(
            f"The column {date_col} must of type datetime64, got {df[date_col].type} instead."
        )

    if season_start not in range(1, 13):
        raise ValueError(
            f"The variable 'season_start' must an integer between 1 and 12, got {season_start} instead."
        )

    out = df.copy()
    season = np.where(
        out[date_col].dt.month >= season_start,
        out[date_col].dt.year,
        out[date_col].dt.year - 1,
    )
    out.insert(0, season_col, season.astype(int))
    return out


def _team_match_split(
    df: pd.DataFrame,
    *,
    home_cols: list[str] = cols.home,
    away_cols: list[str] = cols.away,
    new_cols: list[str] = cols.team_match,
) -> pd.DataFrame:
    """
    Description:
        1st - Check for matching column lengths
        2nd - Extract the columns necessary for the home and away team into a data frame each
        3rd - Unify the column names
        4th - Insert home and away indicator columns to both data frames
        5th - Concatinate both dataframes into one

    Usage location:
        data_creation/prepare.py

    Args:
        df (pd.DataFrame): _description_
        home_cols (list): _description_
        away_cols (list): _description_
        new_cols (list): _description_

    Returns:
        pd.DataFrame: _description_
    """

    if not (len(home_cols) == len(away_cols) == len(new_cols)):
        raise ValueError(
            f"The provided columns have different lengths.\n  home_cols: {len(home_cols)}\n  away_cols: {len(away_cols)}\n  new_cols: {len(new_cols)}"
        )

    out1 = df[home_cols].copy()
    out2 = df[away_cols].copy()
    out1.columns = new_cols
    out2.columns = new_cols
    out1.insert(5, "Home", 1)
    out2.insert(5, "Home", 0)
    out = pd.concat([out1, out2], ignore_index=True)
    return out
