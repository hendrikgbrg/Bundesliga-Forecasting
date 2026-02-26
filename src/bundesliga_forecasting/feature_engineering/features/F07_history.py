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
from bundesliga_forecasting.feature_engineering.F_config import (
    WEIGHTS,
)
from bundesliga_forecasting.feature_engineering.F_utils import (
    create_season_end,
    merge_back,
)

logger = logging.getLogger(__name__)

paths = PATHS
encoding = CSV_ENCODING
cols = COLUMNS
required_cols = [
    cols.season,
    cols.team,
    cols.prev_season_div,
    cols.prev_season_trank,
    cols.prev_season_twins,
    cols.prev_season_tlosses,
    cols.prev_season_tdraws,
    cols.prev_season_tgoaldiff,
    cols.prev_season_tpoint_performance,
]
merge_cols = [
    cols.season,
    cols.team,
    cols.prev_hist_div,
    cols.prev_hist_trank,
    cols.prev_hist_twins,
    cols.prev_hist_tdraws,
    cols.prev_hist_tlosses,
    cols.prev_hist_tgoaldiff,
    cols.prev_hist_tpoint_performance,
]
merge_on = [cols.season, cols.team]


def add_historical_features(
    src_dir: Path = paths.features,
    target_dir: Path = paths.features,
    src_file: str = paths.feature_file,
    target_file: str = paths.feature_file,
) -> None:
    logger.info("Adding historical features to the DataFrame...")
    ensure_dir([src_dir, target_dir], ["src", "target"])

    input_path = src_dir / src_file
    output_path = target_dir / target_file

    df = read_csv(input_path)
    df = df_sort(df, sort_cols=[cols.season, cols.div, cols.date])
    season_end = create_season_end(df, required_cols)
    season_end = _compute_history(season_end)
    df = merge_back(df, season_end, merge_cols=merge_cols, merge_on=merge_on)

    save_to_csv(df, output_path)


##############################################################


def _compute_history(season_end: pd.DataFrame) -> pd.DataFrame:
    pre_hist_map = {
        cols.prev_hist_div: cols.prev_season_div,
        cols.prev_hist_trank: cols.prev_season_trank,
        cols.prev_hist_twins: cols.prev_season_twins,
        cols.prev_hist_tlosses: cols.prev_season_tlosses,
        cols.prev_hist_tdraws: cols.prev_season_tdraws,
        cols.prev_hist_tgoaldiff: cols.prev_season_tgoaldiff,
        cols.prev_hist_tpoint_performance: cols.prev_season_tpoint_performance,
    }

    for hist_col, pre_col in pre_hist_map.items():
        check_columns(season_end, [pre_col])
        season_end[hist_col] = (
            season_end.groupby(cols.team, sort=False)[pre_col]
            # .shift(1, fill_value=0)
            # .groupby(season_end[cols.team], sort=False)
            .ewm(alpha=WEIGHTS.history, adjust=False)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return season_end
