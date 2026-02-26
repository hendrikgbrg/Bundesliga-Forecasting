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
    grouped_aggregate,
    produce_outcome_series,
)

logger = logging.getLogger(__name__)
paths = PATHS
encoding = CSV_ENCODING
cols = COLUMNS


def add_momentum(
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
    df = _add_streak(df)
    df = _add_rolling_point_ratio(df)
    df = _add_rolling_goaldiff_ratio(df)

    save_to_csv(df, output_path)


#################################################################


def _add_streak(df: pd.DataFrame) -> pd.DataFrame:

    ## Internal functions ##
    def _calculate_streaks(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        outcome_series = produce_outcome_series(df)
        df["_result"] = outcome_series.wins - outcome_series.losses
        reset = (df["_result"] == 0) | (
            df["_result"] != df.groupby(group_cols)["_result"].shift(1)
        )
        df["_streak_id"] = reset.groupby([df[col] for col in group_cols]).cumsum()
        df["_streak"] = df.groupby(group_cols + ["_streak_id"]).cumcount() + 1
        df["_streak"] *= df["_result"]
        df[cols.prev_win_streak] = df["_streak"].clip(lower=0)
        df[cols.prev_loss_streak] = df["_streak"].clip(upper=0)
        df = df.drop(columns=["_result", "_streak_id", "_streak"])
        return df

    def _shift_streak_to_prev_match(
        df: pd.DataFrame, group_cols: list[str]
    ) -> pd.DataFrame:
        df[cols.prev_win_streak] = df.groupby(group_cols)[cols.prev_win_streak].shift(
            1, fill_value=0
        )
        df[cols.prev_loss_streak] = df.groupby(group_cols)[cols.prev_loss_streak].shift(
            1, fill_value=0
        )
        return df

    ## Main function ##
    logger.info("Calculating streaks...")
    check_columns(df, [cols.season, cols.team, cols.points])
    group_cols = [cols.season, cols.team]
    df = _calculate_streaks(df, group_cols)
    df = _shift_streak_to_prev_match(df, group_cols)

    return df


def _add_rolling_point_ratio(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating rolling win-loss-rate...")
    check_columns(df, [cols.season, cols.team, cols.points, cols.goalsf, cols.goalsa])

    group_cols = [cols.season, cols.team]
    group_keys = [df[col] for col in group_cols]

    outcome_series = produce_outcome_series(df)

    wins = grouped_aggregate(
        outcome_series.wins, group_keys, window=WEIGHTS.rolling, shift=1
    )
    draws = grouped_aggregate(
        outcome_series.draws, group_keys, window=WEIGHTS.rolling, shift=1
    )
    games = grouped_aggregate(
        outcome_series.games,
        group_keys,
        window=WEIGHTS.rolling,
        shift=1,
        clip_lower=1,
    )

    df[cols.prev_rolling_point_ratio] = (3 * wins + draws) / games

    return df


def _add_rolling_goaldiff_ratio(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating rolling goaldiff-rate...")

    group_cols = [cols.season, cols.team]
    group_keys = [df[col] for col in group_cols]

    outcome_series = produce_outcome_series(df)

    goalsf = grouped_aggregate(
        outcome_series.goalsf, group_keys, window=WEIGHTS.rolling, shift=1
    )
    goalsa = grouped_aggregate(
        outcome_series.goalsa, group_keys, window=WEIGHTS.rolling, shift=1
    )
    games = grouped_aggregate(
        outcome_series.games,
        group_keys,
        window=WEIGHTS.rolling,
        shift=1,
        clip_lower=1,
    )
    df[cols.prev_rolling_goaldiff_ratio] = (goalsf - goalsa) / games

    return df


# def _add_total_rank_performance(df: pd.DataFrame) -> pd.DataFrame:
#     logger.info("Calculating seasonal total rank performance...")
#     check_columns(df, [cols.prev_trank, cols.post_trank])

#     df[cols.prev_trank_performance] = 1 - ((df[cols.prev_trank] - 1) / 35)
#     df[cols.post_trank_performance] = 1 - ((df[cols.post_trank] - 1) / 35)
#     return dd


# def _add_rolling_win_ratio(df: pd.DataFrame) -> pd.DataFrame:
#     logger.info("Calculating rolling win-loss-rate...")
#     check_columns(df, [cols.season, cols.team, cols.points, cols.goalsf, cols.goalsa])

#     group_cols = [cols.season, cols.team]
#     group_keys = [df[col] for col in group_cols]

#     outcome_series = produce_outcome_series(df)

#     wins = grouped_aggregate(
#         outcome_series.wins, group_keys, window=WEIGHTS.rolling, shift=1
#     )
#     draws = grouped_aggregate(
#         outcome_series.draws, group_keys, window=WEIGHTS.rolling, shift=1
#     )
#     games = grouped_aggregate(
#         outcome_series.games,
#         group_keys,
#         window=WEIGHTS.rolling,
#         shift=1,
#         clip_lower=1,
#     )

#     df[cols.prev_win_ratio] = (wins + draws * DRAW_VALUE) / games

#     return df


# def _add_rolling_goal_superiority(df: pd.DataFrame) -> pd.DataFrame:
#     logger.info("Calculating rolling goal superiority...")
#     check_columns(df, [cols.season, cols.team, cols.goalsf, cols.goalsa])

#     group_cols = [cols.season, cols.team]
#     group_keys = [df[col] for col in group_cols]

#     outcome_series = produce_outcome_series(df)

#     numerator = outcome_series.goalsf - outcome_series.goalsa
#     denominator = (outcome_series.goalsf + outcome_series.goalsa).clip(lower=1)
#     goal_superiority = numerator / denominator

#     df[cols.prev_goal_superiority] = grouped_aggregate(
#         goal_superiority,
#         group_keys,
#         window=WEIGHTS.rolling,
#         shift=1,
#         transformer="mean",
#     )

#     return df
