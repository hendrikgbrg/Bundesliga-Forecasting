import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bundesliga_forecasting.BL_config import COLUMNS, CSV_ENCODING, PATHS
from bundesliga_forecasting.BL_utils import (
    check_columns,
    df_sort,
    ensure_dir,
    read_csv,
    save_to_csv,
)
from bundesliga_forecasting.feature_engineering.F_config import (
    ZONES,
)

logger = logging.getLogger(__name__)
paths = PATHS
encoding = CSV_ENCODING
cols = COLUMNS


def add_season_performance(
    src_dir: Path = paths.features,
    target_dir: Path = paths.features,
    src_file: str = paths.feature_file,
    target_file: str = paths.feature_file,
) -> None:
    logger.info("Adding performance-features to the DataFrame...")
    ensure_dir([src_dir, target_dir], ["src", "target"])
    input_path = src_dir / src_file
    output_path = target_dir / target_file

    df = read_csv(input_path)
    df = df_sort(df, sort_cols=[cols.season, cols.div, cols.date])
    df = _add_zones(df)
    df = _add_total_point_performance(df)

    save_to_csv(df, output_path)


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


def _add_total_point_performance(df: pd.DataFrame) -> pd.DataFrame:
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

    df[cols.prev_tpoint_performance] = 1 - (
        (df[cols.prev_max_tpoints] - df[cols.prev_tpoints])
        / np.maximum(df[cols.prev_max_tpoints] - df[cols.prev_min_tpoints], 1)
    )
    df[cols.post_tpoint_performance] = 1 - (
        (df[cols.post_max_tpoints] - df[cols.post_tpoints])
        / np.maximum(df[cols.post_max_tpoints] - df[cols.post_min_tpoints], 1)
    )
    return df
