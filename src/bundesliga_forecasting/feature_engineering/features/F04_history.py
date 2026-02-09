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
from bundesliga_forecasting.feature_engineering.F_utils import (
    grouped_aggregate,
    produce_outcome_series,
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
    season_end = _add_season_outcome_ratios(season_end)
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
        cols.points,
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


def _add_season_outcome_ratios(season_end: pd.DataFrame) -> pd.DataFrame:
    check_columns(season_end, [cols.season, cols.team, cols.points])
    group_cols = [cols.season, cols.team]
    group_keys = [season_end[col] for col in group_cols]

    outcome_series = produce_outcome_series(season_end[cols.points])

    wins = grouped_aggregate(outcome_series.wins, group_keys, window=None, shift=0)
    draws = grouped_aggregate(outcome_series.draws, group_keys, window=None, shift=0)
    losses = grouped_aggregate(
        outcome_series.losses, group_keys, window=None, shift=0, clip_lower=1
    )
    games = grouped_aggregate(
        outcome_series.games, group_keys, window=None, shift=0, clip_lower=1
    )

    season_end[cols.seasonal_win_loss_ratio] = (wins + draws / 3) / losses
    season_end[cols.seasonal_win_ratio] = (wins + draws / 3) / games

    return season_end


def _compute_history(season_end: pd.DataFrame) -> pd.DataFrame:
    pre_hist_map = {
        cols.prev_hist_superiority: cols.post_tgoaldiff,
        cols.prev_hist_trank_performance: cols.post_trank_performance,
        cols.prev_hist_tpoint_performance: cols.post_tpoint_performance,
        cols.prev_hist_win_loss_ratio: cols.seasonal_win_loss_ratio,
        cols.prev_hist_win_ratio: cols.seasonal_win_ratio,
        # cols.hist_tperformance: cols.post_tperformance
    }
    # post_hist_map = {
    #     cols.post_hist_superiority: cols.post_tgoaldiff,
    #     cols.post_hist_trank_performance: cols.post_trank_performance,
    #     cols.post_hist_tpoint_performance: cols.post_tpoint_performance,
    #     # cols.hist_tperformance: cols.post_tperformance
    # }

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

    return season_end


def _merge_back(df: pd.DataFrame, season_end: pd.DataFrame) -> pd.DataFrame:
    merge_cols = [
        cols.season,
        cols.team,
        cols.prev_hist_superiority,
        cols.prev_hist_trank_performance,
        cols.prev_hist_tpoint_performance,
        # cols.post_hist_superiority,
        # cols.post_hist_trank_performance,
        # cols.post_hist_tpoint_performance,
    ]
    merge_on_cols = [cols.season, cols.team]

    check_columns(season_end, merge_cols)
    check_columns(df, merge_on_cols)

    df = df.merge(season_end[merge_cols], on=merge_on_cols, how="left")
    return df
