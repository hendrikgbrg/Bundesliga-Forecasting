import logging
from pathlib import Path

import pandas as pd

from bundesliga_forecasting.BL_config import COLUMNS, CSV_ENCODING, PATHS
from bundesliga_forecasting.BL_utils import (
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
feature_cols = [
    cols.prev_season_trank,
    cols.prev_season_twins,
    cols.prev_season_tdraws,
    cols.prev_season_tlosses,
    cols.prev_season_tgoaldiff,
    cols.prev_season_tpoint_performance,
]
merge_on = [cols.season, cols.team]


def add_relprom_effects(
    src_dir: Path = paths.features,
    target_dir: Path = paths.features,
    src_file: str = paths.feature_file,
    target_file: str = paths.feature_file,
) -> None:
    logger.info("Adding rel-prom effects to the DataFrame...")
    ensure_dir([src_dir, target_dir], ["src", "target"])

    input_path = src_dir / src_file
    output_path = target_dir / target_file

    df = read_csv(input_path)
    df = df_sort(df, sort_cols=[cols.season, cols.div, cols.date])
    df = relprom_effects(df)

    save_to_csv(df, output_path)


##################################################################


def relprom_effects(df: pd.DataFrame) -> pd.DataFrame:
    check_columns(df, [cols.div, cols.prev_season_div] + feature_cols)

    div_diff = df[cols.prev_season_div] - df[cols.div]
    promotion = div_diff == 1
    relegation = div_diff == -1
    for feature in feature_cols:
        df["RelEffect" + feature] = relegation * df[feature]
        df["PromEffect" + feature] = promotion * df[feature]

    return df
