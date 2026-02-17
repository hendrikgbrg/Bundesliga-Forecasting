import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from bundesliga_forecasting.BL_config import COLUMNS, CSV_ENCODING, PATHS, PREDICTORS
from bundesliga_forecasting.BL_utils import (
    check_columns,
    ensure_dir,
    read_csv,
    save_to_csv,
)

logger = logging.getLogger(__name__)

paths = PATHS
preds = PREDICTORS.values()
encoding = CSV_ENCODING
cols = COLUMNS


def select_and_split(
    src_dir: Path = paths.features,
    target_dir: Path = paths.final_poisson,
    src_file: str = paths.diff_file,
    train_file: str = paths.train_file,
    valid_file: str = paths.valid_file,
    test_file: str = paths.test_file,
) -> None:
    logger.info(
        "Selecting features and splitting into train, validation and test datasets..."
    )
    ensure_dir([src_dir, target_dir], ["src", "target"])

    input_path = src_dir / src_file
    output_path_train = target_dir / train_file
    output_path_valid = target_dir / valid_file
    output_path_test = target_dir / test_file

    df = read_csv(input_path)
    df = _select_features(df)
    train_df, valid_df, test_df = _split(df)

    save_to_csv(train_df, output_path_train)
    save_to_csv(valid_df, output_path_valid)
    save_to_csv(test_df, output_path_test)


#################################################################


def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    check_columns(df, [cols.goalsf, cols.date, cols.season] + preds)

    out = df[[cols.goalsf, cols.season, cols.date] + preds]

    return out


def _split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seasons = sorted(df[cols.season].unique())

    train_season = seasons[:-2]
    valid_season = [seasons[-2]]
    test_season = [seasons[-1]]

    train = (
        df[df[cols.season].isin(train_season)]
        .sort_values(cols.date)
        .drop(columns=[cols.season, cols.date])
        .reset_index(drop=True)
    )
    test = (
        df[df[cols.season].isin(test_season)]
        .sort_values(cols.date)
        .drop(columns=[cols.season, cols.date])
        .reset_index(drop=True)
    )
    valid = (
        df[df[cols.season].isin(valid_season)]
        .sort_values(cols.date)
        .drop(columns=[cols.season, cols.date])
        .reset_index(drop=True)
    )

    return train, valid, test
