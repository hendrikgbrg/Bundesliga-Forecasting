import logging
from pathlib import Path

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
)

logger = logging.getLogger(__name__)

paths = PATHS
encoding = CSV_ENCODING
cols = COLUMNS


def add_historical_features(
    src_dir: Path = paths.features,
    target_dir: Path = paths.features,
    src_file: str = paths.f_filename,
    target_file: str = paths.f_filename,
) -> pd.DataFrame:
    logger.info("Adding historical features to the DataFrame...")
    ensure_dir([src_dir, target_dir], ["src", "target"])

    input_path = src_dir / src_file
    output_path = target_dir / target_file

    df = read_csv(input_path)
    df = df_sort(df, sort_cols=[cols.season, cols.div, cols.date])
    calendar = _calendar(df)
    season_end = _season_end(df, calendar)
    season_end = _compute_history(season_end)
    df = _merge_back(df, season_end)

    save_to_csv(df, output_path)
    return df


##############################################################


def _calendar(df: pd.DataFrame) -> pd.DataFrame:
    check_columns(df, [cols.season, cols.team])
    seasons = df[cols.season].drop_duplicates()
    teams = df[cols.team].drop_duplicates()
    calendar = seasons.to_frame().merge(teams.to_frame(), how="cross", sort=False)
    return calendar


def _season_end(df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        cols.season,
        cols.team,
        cols.post_tgoaldiff,
        cols.post_trank_performance,
        cols.post_tpoint_performance,
        # cols.post_tperformance,
    ]
    check_columns(df, required_cols)
    season_end = (
        df[required_cols]
        .groupby([cols.season, cols.team], as_index=False, sort=False)
        .last()
    )
    season_end = calendar.merge(
        season_end, on=[cols.season, cols.team], how="left", sort=False
    ).fillna(0)
    return season_end


def _compute_history(season_end: pd.DataFrame) -> pd.DataFrame:
    pre_hist_map = {
        cols.pre_hist_superiority: cols.post_tgoaldiff,
        cols.pre_hist_trank_performance: cols.post_trank_performance,
        cols.pre_hist_tpoint_performance: cols.post_tpoint_performance,
        # cols.hist_tperformance: cols.post_tperformance
    }
    post_hist_map = {
        cols.post_hist_superiority: cols.post_tgoaldiff,
        cols.post_hist_trank_performance: cols.post_trank_performance,
        cols.post_hist_tpoint_performance: cols.post_tpoint_performance,
        # cols.hist_tperformance: cols.post_tperformance
    }

    for hist_col, pre_col in pre_hist_map.items():
        check_columns(season_end, [pre_col])
        season_end[hist_col] = (
            season_end.groupby(cols.team, sort=False)[pre_col]
            .shift(1, fill_value=0)
            # .groupby(season_end[cols.team])
            .ewm(alpha=EWMA_DECAY.history, adjust=False)
            .mean()
            .reset_index(level=0, drop=True)
        )

    for hist_col, post_col in post_hist_map.items():
        check_columns(season_end, [post_col])
        season_end[hist_col] = (
            season_end.groupby(cols.team, sort=False)[post_col]
            # .groupby(season_end[cols.team])
            .ewm(alpha=EWMA_DECAY.history, adjust=False)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return season_end


def _merge_back(df: pd.DataFrame, season_end: pd.DataFrame) -> pd.DataFrame:
    merge_cols = [
        cols.season,
        cols.team,
        cols.pre_hist_superiority,
        cols.pre_hist_trank_performance,
        cols.pre_hist_tpoint_performance,
        cols.post_hist_superiority,
        cols.post_hist_trank_performance,
        cols.post_hist_tpoint_performance,
    ]
    check_columns(season_end, merge_cols)
    check_columns(df, [cols.season, cols.team])

    df = df.merge(season_end[merge_cols], on=[cols.season, cols.team], how="left")
    return df
