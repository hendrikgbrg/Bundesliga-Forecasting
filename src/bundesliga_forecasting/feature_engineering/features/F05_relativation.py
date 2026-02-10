import logging
from pathlib import Path

import pandas as pd

from bundesliga_forecasting.BL_config import COLUMNS, CSV_ENCODING, PATHS, PREDICTORS
from bundesliga_forecasting.BL_utils import (
    check_columns,
    ensure_dir,
    read_csv,
    save_to_csv,
)

logger = logging.getLogger(__name__)

paths = PATHS
preds = PREDICTORS.values()
encoding = CSV_ENCODING
cols = COLUMNS


def compute_relative_features(
    src_dir: Path = paths.features,
    target_dir: Path = paths.features,
    src_file: str = paths.f_filename,
    target_file: str = paths.r_filename,
) -> pd.DataFrame:
    logger.info("Computing relative features...")
    ensure_dir([src_dir, target_dir], ["src", "target"])

    input_path = src_dir / src_file
    output_path = target_dir / target_file

    df = read_csv(input_path)
    df = _df_split_merge(df)
    save_to_csv(df, output_path)
    return df


#################################################################


def _df_split_merge(df: pd.DataFrame) -> pd.DataFrame:
    match_keys = [cols.season, cols.div, cols.date]
    sites = ["Home", "Away"]
    teams = [site + "Team" for site in sites]
    suffixes: tuple[str, str] = (f"_{sites[0]}", f"_{sites[1]}")
    columns = (
        match_keys
        + [
            cols.team,
            cols.opp,
            cols.goalsf,
            cols.goalsa,
        ]
        + preds
    )

    check_columns(df, columns)

    home_in_rename_map = {cols.team: teams[0], cols.opp: teams[1]}
    away_in_rename_map = {cols.team: teams[1], cols.opp: teams[0]}
    home_out_rename_map = {value: key for key, value in home_in_rename_map.items()}
    away_out_rename_map = {value: key for key, value in away_in_rename_map.items()}

    home_in, away_in = _home_away_input_split(
        df, columns, home_rename=home_in_rename_map, away_rename=away_in_rename_map
    )

    logger.info("Merging the home and away DataFrame for feature relativation...")

    merged = home_in.merge(
        away_in,
        on=match_keys + teams,
        suffixes=suffixes,
        how="inner",
        validate="one_to_one",
    )
    logger.info("Calculating the differences between home and away feature values...")
    for pred in preds:
        merged[pred] = merged[f"{pred}{suffixes[0]}"] - merged[f"{pred}{suffixes[1]}"]

    home_out, away_out = _home_away_output_split(
        merged,
        home_columns=match_keys + teams + preds,
        away_columns=match_keys + teams[::-1] + preds,
        home_rename=home_out_rename_map,
        away_rename=away_out_rename_map,
    )

    away_out[preds] *= -1

    logger.info("Reattachment of home and away DataFrames...")
    df = (
        pd.concat([home_out, away_out], ignore_index=True)
        .sort_values([cols.season, cols.div, cols.date])
        .reset_index(drop=True)
    )

    return df


def _home_away_input_split(
    df: pd.DataFrame,
    columns: list[str],
    *,
    home_rename: dict[str, str],
    away_rename: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(
        "Splitting the original DataFrame into home and away perspectives for feature relativation..."
    )

    home_in = df.loc[df[cols.home] == 1, columns].rename(columns=home_rename)
    away_in = df.loc[df[cols.home] == 0, columns].rename(columns=away_rename)
    return home_in, away_in


def _home_away_output_split(
    merged: pd.DataFrame,
    *,
    home_columns: list[str],
    away_columns: list[str],
    home_rename: dict[str, str],
    away_rename: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Splitting the merged DataFrame after feature relativation...")
    home_out = merged[home_columns].rename(columns=home_rename)
    away_out = merged[away_columns].rename(columns=away_rename)
    return home_out, away_out
