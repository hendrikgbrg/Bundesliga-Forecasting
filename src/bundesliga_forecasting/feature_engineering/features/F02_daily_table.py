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
from bundesliga_forecasting.feature_engineering.F_config import (
    MATCH_COLS,
    POST_RANK_COLS,
    PREV_RANK_COLS,
)

logger = logging.getLogger(__name__)

paths = PATHS
encoding = CSV_ENCODING
cols = COLUMNS

# global variables
rank_group_by = [cols.season, cols.div, cols.date]
RANK_COLS = PREV_RANK_COLS + POST_RANK_COLS


def add_daily_comparisons(
    src_dir: Path = paths.features,
    target_dir: Path = paths.features,
    src_file: str = paths.feature_file,
    target_file: str = paths.feature_file,
) -> None:

    logger.info("Adding season-features to the DataFrame...")
    ensure_dir([src_dir, target_dir], ["src", "target"])

    input_path = src_dir / src_file
    output_path = target_dir / target_file
    required_cols = (
        [cols.season, cols.div, cols.date, cols.team] + MATCH_COLS + RANK_COLS
    )

    df = read_csv(input_path)
    check_columns(df, required_cols)

    daily_tables = _create_daily_tables(df)
    daily_tables = _compute_ranks(daily_tables)
    daily_tables = _add_table_extrema(daily_tables)
    # save_to_csv(daily_tables, target_dir / paths.daily_tables_file)
    df = _merge_back(df, daily_tables)
    save_to_csv(df, output_path)


#########################################################################################################


def _create_daily_tables(df: pd.DataFrame) -> pd.DataFrame:

    ## Internal function ##
    def _create_calendar(df: pd.DataFrame) -> pd.DataFrame:

        logger.info("Creating the season team-date calendar...")
        check_columns(df, [cols.season, cols.div, cols.date, cols.team])
        dates_df = (
            df[[cols.season, cols.div, cols.date]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        teams_df = (
            df[[cols.season, cols.div, cols.team]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        calendar = dates_df.merge(teams_df, on=[cols.season, cols.div], how="inner")

        return calendar

    ## Main function ##
    logger.info("Creating the season snap from the season team-date calendar...")
    check_columns(
        df, [cols.season, cols.div, cols.date, cols.team] + MATCH_COLS + RANK_COLS
    )
    calendar = _create_calendar(df)
    daily_tables = calendar.merge(
        df[[cols.season, cols.div, cols.date, cols.team] + MATCH_COLS + RANK_COLS],
        on=[cols.season, cols.div, cols.date, cols.team],
        how="left",
    )
    return daily_tables


def _sort_ffill_rank(
    daily_tables: pd.DataFrame, rank_cols: list[str], out_col: str
) -> pd.DataFrame:

    ## Internal functions ##
    def _sort_for_ranking(
        daily_tables: pd.DataFrame,
        rank_cols: list[str],
        ascending: list[bool] | None = None,
    ) -> pd.DataFrame:
        if not ascending:
            ascending_group = [True] * len(rank_group_by)
            ascending_rank = [False] * len(rank_cols)
            ascending = ascending_group + ascending_rank
        daily_tables = daily_tables.sort_values(
            by=rank_group_by + rank_cols, ascending=ascending, kind="mergesort"
        )
        return daily_tables

    def _forward_fill(
        daily_tables: pd.DataFrame, col_pairs: list[tuple[str, str]]
    ) -> pd.DataFrame:
        ffill_by = [cols.season, cols.team]
        for prev_col, post_col in col_pairs:
            daily_tables[post_col] = daily_tables.groupby(ffill_by, sort=False)[
                post_col
            ].ffill()
            daily_tables[prev_col] = daily_tables.groupby(ffill_by, sort=False)[
                post_col
            ].shift(1)

        daily_tables[MATCH_COLS + RANK_COLS] = daily_tables[
            MATCH_COLS + RANK_COLS
        ].fillna(0)
        return daily_tables

    def _rank(
        daily_tables: pd.DataFrame, rank_cols: list[str], out_col: str
    ) -> pd.DataFrame:
        base_factor = 10**3
        rev_rank_cols = rank_cols[::-1]
        daily_tables["_rank_key"] = 0
        for index in range(len(rev_rank_cols)):
            daily_tables["_rank_key"] += (
                daily_tables[rev_rank_cols[index]] * base_factor**index
            )
        daily_tables[out_col] = (
            daily_tables.groupby(rank_group_by, sort=False)["_rank_key"]
            .rank(method="dense", ascending=False)
            .astype(int)
        )
        daily_tables.drop(columns=["_rank_key"], inplace=True)
        daily_tables = daily_tables.reset_index(drop=True)
        return daily_tables

    ## Main function ##
    check_columns(
        daily_tables,
        [cols.season, cols.div, cols.date, cols.team] + MATCH_COLS + RANK_COLS,
    )
    col_pairs = list(zip(PREV_RANK_COLS, POST_RANK_COLS))

    daily_tables = _sort_for_ranking(daily_tables, rank_cols)
    daily_tables = _forward_fill(daily_tables, col_pairs)
    daily_tables = _rank(daily_tables, rank_cols, out_col)

    # daily_tables = daily_tables.sort_values(
    #     by=rank_group_by + rank_cols, ascending=ascending, kind="mergesort"
    # )

    return daily_tables


def _compute_ranks(daily_tables: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating ranks by sorting and grouping in the season-snap...")
    check_columns(
        daily_tables,
        [cols.season, cols.div, cols.date] + PREV_RANK_COLS + POST_RANK_COLS,
    )
    rank_configs = [
        {
            "rank_cols": PREV_RANK_COLS,
            "out_col": cols.prev_rank,
            "out_tcol": cols.prev_trank,
        },
        {
            "rank_cols": POST_RANK_COLS,
            "out_col": cols.post_rank,
            "out_tcol": cols.post_trank,
        },
    ]
    # prev and post match rankings
    for config in rank_configs:
        daily_tables = _sort_ffill_rank(
            daily_tables,
            rank_cols=config["rank_cols"],
            out_col=config["out_col"],
        )
        daily_tables[config["out_tcol"]] = np.where(
            daily_tables[cols.div] == 1,
            daily_tables[config["out_col"]],
            daily_tables[config["out_col"]] + 18,
        )

    # post-match ranks
    # daily_tables = _sort_ffill_rank(
    #     daily_tables,
    #     rank_cols=POST_RANK_COLS,
    #     out_col=cols.post_rank,
    # )
    # daily_tables[cols.post_trank] = np.where(
    #     daily_tables[cols.div] == 1,
    #     daily_tables[cols.post_rank],
    #     daily_tables[cols.post_rank] + 18,
    # )
    return daily_tables


def _add_table_extrema(daily_tables: pd.DataFrame) -> pd.DataFrame:
    logger.info("Determining total point extreme values within the season-snap...")
    check_columns(
        daily_tables, [cols.div, cols.date, cols.prev_tpoints, cols.post_tpoints]
    )
    # prior-match max points
    daily_tables[cols.prev_max_tpoints] = daily_tables.groupby([cols.div, cols.date])[
        cols.prev_tpoints
    ].transform("max")

    # prior-match min points
    daily_tables[cols.prev_min_tpoints] = daily_tables.groupby([cols.div, cols.date])[
        cols.prev_tpoints
    ].transform("min")

    # post-match max points
    daily_tables[cols.post_max_tpoints] = daily_tables.groupby([cols.div, cols.date])[
        cols.post_tpoints
    ].transform("max")

    # post-match min points
    daily_tables[cols.post_min_tpoints] = daily_tables.groupby([cols.div, cols.date])[
        cols.post_tpoints
    ].transform("min")

    return daily_tables


def _merge_back(df: pd.DataFrame, daily_tables: pd.DataFrame) -> pd.DataFrame:
    logger.info("Merging season snap back into original DataFrame...")
    merge_columns = [
        cols.season,
        cols.date,
        cols.team,
        cols.prev_min_tpoints,
        cols.prev_max_tpoints,
        cols.prev_rank,
        cols.prev_trank,
        cols.post_min_tpoints,
        cols.post_max_tpoints,
        cols.post_rank,
        cols.post_trank,
    ]
    on_columns = [cols.season, cols.date, cols.team]
    check_columns(daily_tables, merge_columns)

    df = df.merge(
        daily_tables[merge_columns],
        on=on_columns,
        how="left",
    )

    return df
