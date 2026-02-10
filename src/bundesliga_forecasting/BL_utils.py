from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from bundesliga_forecasting.BL_config import COLUMNS, CSV_ENCODING

logger = logging.getLogger(__name__)
cols = COLUMNS
SortKind = Literal["quicksort", "mergesort", "heapsort", "stable"]


def read_csv(
    input_path: Path,
    *,
    parse_dates: list[str] = [cols.date],
    dayfirst: bool = True,
    encoding: str = CSV_ENCODING,
) -> pd.DataFrame:
    df = pd.read_csv(input_path, encoding=encoding)
    for col in parse_dates:
        df[cols.date] = pd.to_datetime(
            df[cols.date], dayfirst=True, errors="raise", format="mixed"
        )
    return df


def save_to_csv(df: pd.DataFrame, output_path: Path, *, index: bool = False) -> None:
    df.to_csv(output_path, index=index)


def check_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(
            f"The following columns are missing in the DataFrame: {missing}."
        )


def df_sort(
    df: pd.DataFrame, *, sort_cols: list[str], kind: SortKind = "mergesort"
) -> pd.DataFrame:
    check_columns(df, sort_cols)
    df = df.sort_values(sort_cols, kind=kind)
    return df


def ensure_dir(paths: list[Path], dir_types: list[Literal["src", "target"]]) -> None:
    if len(paths) != len(dir_types):
        raise ValueError(
            "The list objects 'paths' and 'dir_types' have to be of the same length."
        )
    for path, dir_type in zip(paths, dir_types):
        if not path.exists():
            if dir_type == "src":
                raise FileNotFoundError(f"Source directory does not exist: {path}")
            if dir_type == "target":
                path.mkdir(parents=True, exist_ok=True)
                logger.info("Created directory: %s", path)
        else:
            if not path.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {path}.")
            if dir_type == "src":
                if not any(path.iterdir()):
                    raise FileNotFoundError("Source directory is empty.")
