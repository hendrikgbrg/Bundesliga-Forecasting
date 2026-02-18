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

logger = logging.getLogger(__name__)

paths = PATHS
encoding = CSV_ENCODING
cols = COLUMNS


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
    calendar = _calendar(df)
    season_end = _season_end(df, calendar)
    season_end = _add_season_goal_superiority(season_end)
    season_end = _compute_history(season_end)
    df = _merge_back(df, season_end)

    save_to_csv(df, output_path)


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
        cols.post_tgoalsf,
        cols.post_tgoalsa,
        cols.post_trank_performance,
        cols.post_tpoint_performance,
        cols.seasonal_win_loss_ratio,
        cols.seasonal_win_ratio,
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


def _add_season_goal_superiority(season_end: pd.DataFrame) -> pd.DataFrame:
    check_columns(season_end, [cols.post_tgoalsf, cols.post_tgoalsa])

    tgoalsf = season_end[cols.post_tgoalsf]
    tgoalsa = season_end[cols.post_tgoalsa]
    numerator = tgoalsf - tgoalsa
    denominator = (tgoalsf + tgoalsa).clip(lower=1)
    seasonal_goal_superiority = numerator / denominator

    season_end[cols.seasonal_goal_superiority] = seasonal_goal_superiority

    return season_end


def _compute_history(season_end: pd.DataFrame) -> pd.DataFrame:
    pre_hist_map = {
        cols.prev_hist_goal_superiority: cols.seasonal_goal_superiority,
        cols.prev_hist_trank_performance: cols.post_trank_performance,
        cols.prev_hist_tpoint_performance: cols.post_tpoint_performance,
        cols.prev_hist_win_loss_ratio: cols.seasonal_win_loss_ratio,
        cols.prev_hist_win_ratio: cols.seasonal_win_ratio,
    }

    for hist_col, pre_col in pre_hist_map.items():
        check_columns(season_end, [pre_col])
        season_end[hist_col] = (
            season_end.groupby(cols.team, sort=False)[pre_col]
            .shift(1, fill_value=0)
            .groupby(season_end[cols.team], sort=False)
            .ewm(alpha=WEIGHTS.history, adjust=False)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return season_end


def _merge_back(df: pd.DataFrame, season_end: pd.DataFrame) -> pd.DataFrame:
    merge_cols = [
        cols.season,
        cols.team,
        cols.seasonal_goal_superiority,
        cols.prev_hist_goal_superiority,
        cols.prev_hist_trank_performance,
        cols.prev_hist_tpoint_performance,
        cols.prev_hist_win_loss_ratio,
        cols.prev_hist_win_ratio,
    ]
    merge_on_cols = [cols.season, cols.team]

    check_columns(season_end, merge_cols)
    check_columns(df, merge_on_cols)

    df = df.merge(season_end[merge_cols], on=merge_on_cols, how="left")
    return df
