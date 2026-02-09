import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bundesliga_forecasting.BL_config import CSV_ENCODING, PATHS
from bundesliga_forecasting.BL_utils import (
    check_columns,
    df_sort,
    ensure_dir,
    read_csv,
    save_to_csv,
)
from bundesliga_forecasting.feature_engineering.F_config import (
    COLUMNS,
    EWMA_DECAY,
    ZONES,
)
from bundesliga_forecasting.feature_engineering.F_utils import (
    grouped_aggregate,
    produce_outcome_series,
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
    logger.info("Adding performance-features to the DataFrame...")
    ensure_dir([src_dir, target_dir], ["src", "target"])

    input_path = src_dir / src_file
    output_path = target_dir / target_file

    df = read_csv(input_path)
    df = df_sort(df, sort_cols=[cols.season, cols.div, cols.date])
    df = _add_zones(df)
    df = _total_rank_performance(df)
    df = _total_point_performance(df)
    # df = _cum_performance(df)
    df = _rolling_win_loss_ratio(df)
    df = _rolling_win_ratio(df)
    df = _ewma_superiority(df)

    save_to_csv(df, output_path)
    return df


#################################################################


def _add_zones(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating table zones...")
    check_columns(df, [cols.prev_rank])

    df[cols.zone] = pd.cut(
        df[cols.prev_rank],
        bins=ZONES.bins,
        labels=ZONES.labels,
        right=True,
        include_lowest=True,
    )
    return df


def _total_rank_performance(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating seasonal total rank performance...")
    check_columns(df, [cols.prev_trank, cols.post_trank])

    df[cols.prev_trank_performance] = 1 - 2 * ((df[cols.prev_trank] - 1) / 35)
    df[cols.post_trank_performance] = 1 - 2 * ((df[cols.post_trank] - 1) / 35)
    return df


def _total_point_performance(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating seasonal total point performance...")
    check_columns(
        df,
        [
            cols.prev_max_tpoints,
            cols.prev_min_tpoints,
            cols.prev_tpoints,
            cols.post_max_tpoints,
            cols.post_min_tpoints,
            cols.post_tpoints,
        ],
    )

    df[cols.prev_tpoint_performance] = 1 - 2 * (
        (df[cols.prev_max_tpoints] - df[cols.prev_tpoints])
        / np.maximum(df[cols.prev_max_tpoints] - df[cols.prev_min_tpoints], 1)
    )
    df[cols.post_tpoint_performance] = 1 - 2 * (
        (df[cols.post_max_tpoints] - df[cols.post_tpoints])
        / np.maximum(df[cols.post_max_tpoints] - df[cols.post_min_tpoints], 1)
    )
    return df


# def _cum_performance(df: pd.DataFrame) -> pd.DataFrame:
#     df[cols.prev_tperformance] = 0.5 * (
#         df[cols.prev_tpoint_performance] + df[cols.prev_rank_performance]
#     )
#     df[cols.post_tperformance] = 0.5 * (
#         df[cols.post_tpoint_performance] + df[cols.post_rank_performance]
#     )
#     return df


def _rolling_win_loss_ratio(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating rolling win-loss-ratio...")
    check_columns(df, [cols.season, cols.team, cols.points])

    group_cols = [cols.season, cols.team]
    group_keys = [df[col] for col in group_cols]

    outcome_series = produce_outcome_series(df[cols.points])

    wins = grouped_aggregate(
        outcome_series.wins, group_keys, window=EWMA_DECAY.rolling, shift=1
    )
    draws = grouped_aggregate(
        outcome_series.draws, group_keys, window=EWMA_DECAY.rolling, shift=1
    )
    losses = grouped_aggregate(
        outcome_series.losses,
        group_keys,
        window=EWMA_DECAY.rolling,
        shift=1,
        clip_lower=1,
    )

    df[cols.prev_win_loss_ratio] = (wins + draws / 3) / losses

    return df


def _rolling_win_ratio(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating rolling win-loss-rate...")
    check_columns(df, [cols.season, cols.team, cols.points])

    group_cols = [cols.season, cols.team]
    group_keys = [df[col] for col in group_cols]

    outcome_series = produce_outcome_series(df[cols.points])

    wins = grouped_aggregate(
        outcome_series.wins, group_keys, window=EWMA_DECAY.rolling, shift=1
    )
    draws = grouped_aggregate(
        outcome_series.draws, group_keys, window=EWMA_DECAY.rolling, shift=1
    )
    games = grouped_aggregate(
        outcome_series.games,
        group_keys,
        window=EWMA_DECAY.rolling,
        shift=1,
        clip_lower=1,
    )

    df[cols.prev_win_ratio] = (wins + draws / 3) / games

    return df


def _ewma_superiority(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating seasonally weighted superiority...")
    check_columns(df, [cols.season, cols.team, cols.goaldiff])

    df[cols.prev_goaldiff] = df.groupby([cols.season, cols.team])[cols.goaldiff].shift(
        1, fill_value=0
    )
    df[cols.prev_superiority] = (
        df.groupby([cols.season, cols.team], sort=False)[cols.prev_goaldiff]
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
