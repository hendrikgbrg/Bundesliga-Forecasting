import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bundesliga_forecasting.feature_engineering.feature_config import COLUMNS
from bundesliga_forecasting.project_config import CSV_ENCODING, PATHS
from bundesliga_forecasting.project_utils import (
    check_columns,
    ensure_dir,
    read_csv,
    save_to_csv,
)

logger = logging.getLogger(__name__)

cols = COLUMNS
paths = PATHS
encoding = CSV_ENCODING


def add_score_features(
    src_dir: Path = paths.prepared,
    target_dir: Path = paths.features,
    src_file: str = paths.p_filename,
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
    logger.info("Adding score-features to the DataFrame...")
    ensure_dir([src_dir, target_dir], ["src", "target"])

    input_path = src_dir / src_file
    output_path = target_dir / target_file
    required_cols = [cols.season, cols.team, cols.goalsf, cols.goalsa]

    df = read_csv(input_path)
    check_columns(df, required_cols)

    df = _add_goaldiff(df)
    df = _add_points(df)
    df = _add_post_tgoalsf(df)
    df = _add_post_tgoalsa(df)
    df = _add_post_tgoaldiff(df)
    df = _add_post_tpoints(df)
    df = _add_pre_tgoalsf(df)
    df = _add_pre_tgoalsa(df)
    df = _add_pre_tgoaldiff(df)
    df = _add_pre_tpoints(df)

    save_to_csv(df, output_path)
    return df


#######################################################


# match score
def _add_goaldiff(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating goal differences on team-match level...")
    df[cols.goaldiff] = df[cols.goalsf] - df[cols.goalsa]
    return df


def _add_points(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating points on team-match level...")
    df[cols.points] = np.where(
        df[cols.goaldiff] > 0,
        3,
        np.where(df[cols.goaldiff] == 0, 1, 0),
    )
    return df


# post match total score
def _add_post_tgoalsf(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cumulating total post-match goals scored for...")
    df[cols.post_tgoalsf] = df.groupby([cols.season, cols.team])[cols.goalsf].cumsum()
    return df


def _add_post_tgoalsa(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cumulating total post-match goals scored gainst...")
    df[cols.post_tgoalsa] = df.groupby([cols.season, cols.team])[cols.goalsa].cumsum()
    return df


def _add_post_tgoaldiff(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cumulating total post-match goal differences...")
    df[cols.post_tgoaldiff] = df[cols.post_tgoalsf] - df[cols.post_tgoalsa]
    return df


def _add_post_tpoints(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cumulating total post-match points...")
    df[cols.post_tpoints] = df.groupby([cols.season, cols.team])[cols.points].cumsum()
    return df


# prior match total score
def _add_pre_tgoalsf(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cumulating total pre-match goals scored for...")
    df[cols.pre_tgoalsf] = df[cols.post_tgoalsf] - df[cols.goalsf]
    return df


def _add_pre_tgoalsa(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cumulating total pre-match goals scored against...")
    df[cols.pre_tgoalsa] = df[cols.post_tgoalsa] - df[cols.goalsa]
    return df


def _add_pre_tgoaldiff(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cumulating total pre-match goal differences...")
    df[cols.pre_tgoaldiff] = df[cols.post_tgoaldiff] - df[cols.goaldiff]
    return df


def _add_pre_tpoints(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cumulating total pre-match points...")
    df[cols.pre_tpoints] = df[cols.post_tpoints] - df[cols.points]
    return df
