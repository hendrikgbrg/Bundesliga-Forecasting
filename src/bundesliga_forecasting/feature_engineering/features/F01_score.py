import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bundesliga_forecasting.BL_config import COLUMNS, CSV_ENCODING, PATHS
from bundesliga_forecasting.BL_utils import (
    check_columns,
    ensure_dir,
    read_csv,
    save_to_csv,
)
from bundesliga_forecasting.feature_engineering.F_utils import (
    produce_outcome_series,
)

logger = logging.getLogger(__name__)

cols = COLUMNS
paths = PATHS
encoding = CSV_ENCODING


def add_score_features(
    src_dir: Path = paths.prepared,
    target_dir: Path = paths.features,
    src_file: str = paths.prepared_file,
    target_file: str = paths.feature_file,
) -> None:

    logger.info("Adding score-features to the DataFrame...")
    ensure_dir([src_dir, target_dir], ["src", "target"])

    input_path = src_dir / src_file
    output_path = target_dir / target_file
    required_cols = [cols.season, cols.team, cols.goalsf, cols.goalsa]

    df = read_csv(input_path)
    check_columns(df, required_cols)

    df = _add_match_scores(df)
    df = _add_cum_post_match_scores(df)
    df = _add_cum_prev_match_scores(df)

    save_to_csv(df, output_path)


#######################################################


# match score
def _add_match_scores(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating scores on team-match level...")
    df[cols.goaldiff] = df[cols.goalsf] - df[cols.goalsa]
    df[cols.points] = np.where(
        df[cols.goaldiff] > 0,
        3,
        np.where(df[cols.goaldiff] == 0, 1, 0),
    )
    outcome = produce_outcome_series(df)
    df["win"] = outcome.wins
    df["loss"] = outcome.losses
    df["draw"] = outcome.draws
    return df


# post match total score
def _add_cum_post_match_scores(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cumulating post-match scores on team-match level...")
    df[cols.post_tgoalsf] = df.groupby([cols.season, cols.team])[cols.goalsf].cumsum()
    df[cols.post_tgoalsa] = df.groupby([cols.season, cols.team])[cols.goalsa].cumsum()
    df[cols.post_tgoaldiff] = df[cols.post_tgoalsf] - df[cols.post_tgoalsa]
    df[cols.post_tpoints] = df.groupby([cols.season, cols.team])[cols.points].cumsum()
    df[cols.post_twins] = df.groupby([cols.season, cols.team])["win"].cumsum()
    df[cols.post_tlosses] = df.groupby([cols.season, cols.team])["loss"].cumsum()
    df[cols.post_tdraws] = df.groupby([cols.season, cols.team])["draw"].cumsum()
    return df


# prior match total score
def _add_cum_prev_match_scores(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cumulating pre-match scores on team-match level...")
    df[cols.prev_tgoalsf] = df[cols.post_tgoalsf] - df[cols.goalsf]
    df[cols.prev_tgoalsa] = df[cols.post_tgoalsa] - df[cols.goalsa]
    df[cols.prev_tgoaldiff] = df[cols.post_tgoaldiff] - df[cols.goaldiff]
    df[cols.prev_tpoints] = df[cols.post_tpoints] - df[cols.points]
    df[cols.prev_twins] = df[cols.post_twins] - df["win"]
    df[cols.prev_tlosses] = df[cols.post_tlosses] - df["loss"]
    df[cols.prev_tdraws] = df[cols.post_tdraws] - df["draw"]
    df = df.drop(columns=["win", "loss", "draw"])
    return df


def _add_match_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    return df
