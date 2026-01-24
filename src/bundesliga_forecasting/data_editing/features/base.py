from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


def add_season(
    df: pd.DataFrame,
    *,
    date_col: str = "Date",
    season_col: str = "Season",
    season_start: int = 7,
) -> pd.DataFrame:
    """
    Description:

    Args:
        df (pd.DataFrame): _description_
        date_col (str, optional): _description_. Defaults to "Date".
        season_col (str, optional): _description_. Defaults to "Season".
        eason_start (int, optional): _description_. Defaults to 7.

    Raises:
        KeyError: _description_

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


def team_match_split(
    df: pd.DataFrame, *, home_cols: list[str], away_cols: list[str], new_cols: list[str]
) -> pd.DataFrame:
    """
    Description:
    ------------



    Args:
    -----

        df (pd.DataFrame): _description_
        home_cols (list): _description_
        away_cols (list): _description_
        new_cols (list): _description_

    Returns:
    --------

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
