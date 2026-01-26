from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

# ----------
#  Cleaning
# ----------


def extract_columns(
    lines: list[str],
    col_names: list[str],
) -> list[str]:
    """
    Description:
        1st -

    Usage location:
        data_creation/io/clean_csv_directory.py

    Args:
        df (pd.DataFrame): _description_
        home_cols (list): _description_
        away_cols (list): _description_
        new_cols (list): _description_

    Returns:
        pd.DataFrame: _description_
    """
    header = lines[0].strip().split(",")

    try:
        col_indices = [header.index(col) for col in col_names]
    except ValueError as e:
        raise ValueError("Missing required column in CSV header") from e

    rows = []
    for line in lines[1:]:
        fields = line.strip().split(",")

        if not any(field.strip() for field in fields):
            continue

        selected = [fields[idx] for idx in col_indices]
        rows.append(selected)

    return rows


def adjust_team_names(
    df: pd.DataFrame, col_names: list[str], rename_map: dict[str, str]
) -> pd.DataFrame:
    """
    Description:

    Usage location:
        data_creation/io/clean_csv_directory.py

    Args:
        df (pd.DataFrame): _description_
        col_names (list[str]): _description_
        rename_map (dict[str, str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df[col_names] = df[col_names].apply(lambda col: col.str.strip())

    return df.replace(rename_map)


# ---------------
#  Restructuring
# ---------------


def add_season(
    df: pd.DataFrame,
    *,
    date_col: str = "Date",
    season_col: str = "Season",
    season_start: int = 7,
) -> pd.DataFrame:
    """
    Description:

    Usage location:
        data_creation/pipeline.py

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


def team_match_split(
    df: pd.DataFrame, *, home_cols: list[str], away_cols: list[str], new_cols: list[str]
) -> pd.DataFrame:
    """
    Description:
        1st - Check for matching column lengths
        2nd - Extract the columns necessary for the home and away team into a data frame each
        3rd - Unify the column names
        4th - Insert home and away indicator columns to both data frames
        5th - Concatinate both dataframes into one

    Usage locations:
        data_creation/pipeline.py

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


def sort_matches(df: pd.DataFrame, *, sort_cols: list[str]) -> pd.DataFrame:
    """
    Description:

    Usage location:
        data_creation/pipeline.py

    Args:
        df (pd.DataFrame): _description_
        sort_cols (list[str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    missing = [col for col in sort_cols if col not in df.columns]
    if missing:
        raise KeyError(
            "The following columns are missing in the given DataFrame: {missing}"
        )

    return df.sort_values(sort_cols).reset_index(drop=True)
