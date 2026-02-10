from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from bundesliga_forecasting.BL_config import COLUMNS, CSV_ENCODING, PATHS
from bundesliga_forecasting.BL_utils import ensure_dir, save_to_csv
from bundesliga_forecasting.data_structuring.S_config import COLUMNLISTS, RENAME_MAP
from bundesliga_forecasting.data_structuring.S_utils import detect_csv_files

logger = logging.getLogger(__name__)

paths = PATHS
col_lists = COLUMNLISTS
encoding = CSV_ENCODING
cols = COLUMNS


def clean(
    src_dir: Path = paths.raw,
    target_dir: Path = paths.cleaned,
    *,
    encoding: str = encoding,
) -> None:
    """
    Description:
        Step 1 -> Read the lines of each CSV-file in the source directory
        Step 2 -> Create the data frame with required columns & remove empty rows and spaces
        Step 3 -> Adjust team names
        Step 4 -> Save the data frame to the target directory

    Usage location:
        data_creation/pipeline.py

    Args:
        src_dir (Path): _description_
        target_dir (Path): _description_
        col_names (list[str]): _description_
        rename_map (dict[str, str]): _description_
        encoding (str, optional): _description_. Defaults to "latin1".
    """

    logger.info("Starting file cleaning...")

    ensure_dir([src_dir, target_dir], ["src", "target"])

    csv_files = detect_csv_files(src_dir)

    for file in csv_files:
        logger.info("Processing: %s", file.name)
        with open(file, encoding=encoding) as f:
            lines = f.readlines()

            # Step 2:
            rows = _extract_columns(lines)
            df = pd.DataFrame(data=rows, columns=col_lists.raw)
            df[cols.date] = pd.to_datetime(
                df[cols.date], dayfirst=True, errors="raise", format="mixed"
            )

            # Step 3:
            df = _adjust_team_names(df)

            # Step 4:
            save_to_csv(df, target_dir / file.name)

    logger.info(
        f"{len(csv_files)} files cleaned successfully and saved in {target_dir}."
    )


def _extract_columns(
    lines: list[str],
    col_names: list[str] = col_lists.raw,
) -> list[str]:
    """
    Description:
        1st -

    Usage location:
        data_creation/io/clean.py

    Args:
        df (pd.DataFrame): _description_
        home_cols (list): _description_
        away_cols (list): _description_
        new_cols (list): _description_

    Returns:
        pd.DataFrame: _description_
    """

    if len(lines) == 0:
        raise ValueError(
            "List object 'lines' is empty. No columns have been extracted."
        )
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


def _adjust_team_names(
    df: pd.DataFrame,
    team_cols: list[str] = col_lists.team,
    rename_teams: dict[str, str] = RENAME_MAP,
) -> pd.DataFrame:
    """
    Description:

    Usage location:
        data_creation/io/clean.py

    Args:
        df (pd.DataFrame): _description_
        col_names (list[str]): _description_
        rename_map (dict[str, str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df[team_cols] = df[team_cols].apply(lambda col: col.str.strip())

    return df.replace(rename_teams)
