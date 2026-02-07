import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bundesliga_forecasting.feature_engineering.feature_config import (
    COLUMNS,
    EWMA_DECAY,
    ZONES,
)
from bundesliga_forecasting.project_config import CSV_ENCODING, PATHS
from bundesliga_forecasting.project_utils import (
    check_columns,
    df_sort,
    ensure_dir,
    read_csv,
    save_to_csv,
)

logger = logging.getLogger(__name__)

paths = PATHS
encoding = CSV_ENCODING
cols = COLUMNS


def add_performance_features(
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
    ensure_dir([src_dir, target_dir], ["src", "target"])

    input_path = src_dir / src_file
    output_path = target_dir / target_file

    df = read_csv(input_path)
    df = df_sort(df, sort_cols=[cols.season, cols.div, cols.date])
    df = _add_zones(df)
    df = _total_rank_performance(df)
    df = _total_point_performance(df)
    # df = _cum_performance(df)
    df = _rolling_win_loss_rate(df)
    df = _ewma_superiority(df)

    save_to_csv(df, output_path)
    return df


#################################################################


def _add_zones(df: pd.DataFrame) -> pd.DataFrame:
    check_columns(df, [cols.pre_rank])
    df[cols.zone] = pd.cut(
        df[cols.pre_rank],
        bins=ZONES.bins,
        labels=ZONES.labels,
        right=True,
        include_lowest=True,
    )
    return df


def _total_rank_performance(df: pd.DataFrame) -> pd.DataFrame:
    check_columns(df, [cols.pre_trank, cols.post_trank])

    df[cols.pre_trank_performance] = 1 - 2 * ((df[cols.pre_trank] - 1) / 35)
    df[cols.post_trank_performance] = 1 - 2 * ((df[cols.post_trank] - 1) / 35)
    return df


def _total_point_performance(df: pd.DataFrame) -> pd.DataFrame:
    df[cols.pre_tpoint_performance] = 1 - 2 * (
        (df[cols.pre_max_tpoints] - df[cols.pre_tpoints])
        / np.maximum(df[cols.pre_max_tpoints] - df[cols.pre_min_tpoints], 1)
    )
    df[cols.post_tpoint_performance] = 1 - 2 * (
        (df[cols.post_max_tpoints] - df[cols.post_tpoints])
        / np.maximum(df[cols.post_max_tpoints] - df[cols.post_min_tpoints], 1)
    )
    return df


# def _cum_performance(df: pd.DataFrame) -> pd.DataFrame:
#     df[cols.pre_tperformance] = 0.5 * (
#         df[cols.pre_tpoint_performance] + df[cols.pre_rank_performance]
#     )
#     df[cols.post_tperformance] = 0.5 * (
#         df[cols.post_tpoint_performance] + df[cols.post_rank_performance]
#     )
#     return df


def _rolling_win_loss_rate(df: pd.DataFrame) -> pd.DataFrame:
    wins = (df[cols.points] == 3).astype(int)
    draws = (df[cols.points] == 1).astype(int)
    losses = (df[cols.points] == 0).astype(int)

    pre_wins = (
        wins.groupby([df[cols.season], df[cols.team]], sort=False)
        .shift(1, fill_value=0)
        .rolling(EWMA_DECAY.rolling, min_periods=1)
        .sum()
        .reset_index(drop=True)
    )
    pre_draws = (
        draws.groupby([df[cols.season], df[cols.team]], sort=False)
        .shift(1, fill_value=0)
        .rolling(EWMA_DECAY.rolling, min_periods=1)
        .sum()
        .reset_index(drop=True)
    )
    pre_losses = (
        losses.groupby([df[cols.season], df[cols.team]], sort=False)
        .shift(1, fill_value=0)
        .rolling(EWMA_DECAY.rolling, min_periods=1)
        .sum()
        .clip(lower=1)
        .reset_index(drop=True)
    )

    post_wins = (
        wins.groupby([df[cols.season], df[cols.team]], sort=False)
        .rolling(EWMA_DECAY.rolling, min_periods=1)
        .sum()
        .reset_index(drop=True)
    )
    post_draws = (
        draws.groupby([df[cols.season], df[cols.team]], sort=False)
        .rolling(EWMA_DECAY.rolling, min_periods=1)
        .sum()
        .reset_index(drop=True)
    )
    post_losses = (
        losses.groupby([df[cols.season], df[cols.team]], sort=False)
        .rolling(EWMA_DECAY.rolling, min_periods=1)
        .sum()
        .clip(lower=1)
        .reset_index(drop=True)
    )

    df[cols.pre_win_loss_rate] = (pre_wins + 1 / 3 * pre_draws) / pre_losses

    df[cols.post_win_loss_rate] = (post_wins + 1 / 3 * post_draws) / post_losses

    return df


def _ewma_superiority(df: pd.DataFrame) -> pd.DataFrame:
    df[cols.pre_goaldiff] = df[cols.goaldiff].shift(1, fill_value=0)
    df[cols.pre_superiority] = (
        df.groupby([cols.season, cols.team], sort=False)[cols.pre_goaldiff]
        .ewm(alpha=EWMA_DECAY.season, adjust=False)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    df[cols.post_superiority] = (
        df.groupby([cols.season, cols.team], sort=False)[cols.goaldiff]
        .ewm(alpha=EWMA_DECAY.season, adjust=False)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    return df
