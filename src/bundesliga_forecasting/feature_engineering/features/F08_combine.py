import logging
from pathlib import Path

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


def apply_feature_combination(
    src_dir: Path = paths.features,
    target_dir: Path = paths.features,
    src_file: str = paths.feature_file,
    target_file: str = paths.combined_file,
) -> None:
    logger.info("Combining home and away features...")
    ensure_dir([src_dir, target_dir], ["src", "target"])

    input_path = src_dir / src_file
    output_path = target_dir / target_file

    df = read_csv(input_path)
    df = _combine_home_away_features(df)
    df = _select_features(df)
    save_to_csv(df, output_path)


#################################################################


def _combine_home_away_features(df: pd.DataFrame) -> pd.DataFrame:
    match_keys = [cols.season, cols.div, cols.date]
    join_keys_left = match_keys + [cols.team, cols.opp]
    join_keys_right = match_keys + [cols.opp, cols.team]

    check_columns(df, join_keys_left + list(preds))
    combined = df.merge(
        df,
        left_on=join_keys_left,
        right_on=join_keys_right,
        suffixes=("", "_opp"),
        how="inner",
        validate="one_to_one",
    )

    return combined


def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    opp_preds = [
        f"{pred}_opp" for pred in preds if (pred != cols.home) * (pred != cols.div)
    ]
    check_columns(
        df,
        [cols.goalsf, cols.season, cols.div, cols.date, cols.team, cols.opp]
        + preds
        + opp_preds,
    )

    out = df[
        [cols.goalsf, cols.season, cols.div, cols.date, cols.team, cols.opp]
        + preds
        + opp_preds
    ]

    return out
