import logging
from collections.abc import Callable
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
from bundesliga_forecasting.feature_engineering.F_utils import (
    create_season_end,
    merge_back,
    prev_season_value,
)

logger = logging.getLogger(__name__)

paths = PATHS
encoding = CSV_ENCODING
cols = COLUMNS
required_cols = [
    cols.season,
    cols.div,
    cols.team,
    cols.post_tgoalsf,
    cols.post_tgoalsa,
    cols.post_tgoaldiff,
    cols.post_trank,
    cols.post_twins,
    cols.post_tlosses,
    cols.post_tdraws,
    cols.post_tpoint_performance,
]
merge_cols = [
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
merge_on = [cols.season, cols.team]


def add_prev_season_performance(
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
    season_end = _add_prev_season_division(season_end)
    season_end = _add_prev_season_trank(season_end)
    season_end = _add_prev_season_outcomes(season_end)
    season_end = _add_prev_season_tgoaldiff(season_end)
    season_end = _add_prev_season_tpoint_performance(season_end)
    df = merge_back(df, season_end, merge_cols=merge_cols, merge_on=merge_on)

    save_to_csv(df, output_path)


##############################################################


def _add_prev_season_division(season_end: pd.DataFrame) -> pd.DataFrame:

    ## Internal function ##
    def _fallback_function(
        team_season: pd.DataFrame, mask: pd.Series, ref_col: str, new_col: str
    ) -> pd.Series:
        return team_season.loc[mask, ref_col]

    ## Main function ##
    season_end = _add_prev_season_feature(
        season_end,
        ref_col=cols.div,
        new_col=cols.prev_season_div,
        required_cols=[cols.season, cols.div, cols.team],
        fallback_function=_fallback_function,
        fillval=3,
    )

    return season_end


def _add_prev_season_trank(season_end: pd.DataFrame) -> pd.DataFrame:

    ## Internal function ##
    def _fallback_function(
        team_season: pd.DataFrame, mask: pd.Series, ref_col: str, new_col: str
    ) -> pd.Series:
        series = team_season.loc[mask, cols.prev_season_div]
        return (series - 1) * 18 + 1

    ## Main function ##
    season_end = _add_prev_season_feature(
        season_end,
        ref_col=cols.post_trank,
        new_col=cols.prev_season_trank,
        required_cols=[cols.season, cols.div, cols.team, cols.post_trank],
        fallback_function=_fallback_function,
        fillval=37,
    )

    return season_end


def _add_prev_season_outcomes(season_end: pd.DataFrame) -> pd.DataFrame:

    ## Internal function ##
    def _fallback_function(
        team_season: pd.DataFrame, mask: pd.Series, ref_col: str, new_col: str
    ) -> pd.Series:
        return team_season.loc[mask, new_col].fillna(0)

    ## Main function ##
    ref_cols = [cols.post_twins, cols.post_tdraws, cols.post_tlosses]
    new_cols = [
        cols.prev_season_twins,
        cols.prev_season_tdraws,
        cols.prev_season_tlosses,
    ]
    for ref_col, new_col in list(zip(ref_cols, new_cols)):
        season_end = _add_prev_season_feature(
            season_end,
            ref_col=ref_col,
            new_col=new_col,
            required_cols=[cols.season, cols.div, cols.team, ref_col],
            fallback_function=_fallback_function,
            fillval=0,
        )

    return season_end


def _add_prev_season_tgoaldiff(season_end: pd.DataFrame) -> pd.DataFrame:

    ## Internal function ##
    def _fallback_function(
        team_season: pd.DataFrame, mask: pd.Series, ref_col: str, new_col: str
    ) -> pd.Series:
        return team_season[new_col].fillna(0)

    ## Main function ##
    season_end = _add_prev_season_feature(
        season_end,
        ref_col=cols.post_tgoaldiff,
        new_col=cols.prev_season_tgoaldiff,
        required_cols=[cols.season, cols.div, cols.team, cols.post_tgoaldiff],
        fallback_function=_fallback_function,
        fillval=0,
    )

    return season_end


def _add_prev_season_tpoint_performance(season_end: pd.DataFrame) -> pd.DataFrame:

    ## Internal function ##
    def _fallback_function(
        team_season: pd.DataFrame, mask: pd.Series, ref_col: str, new_col: str
    ) -> pd.Series:
        return team_season[new_col].fillna(0)

    ## Main function ##
    season_end = _add_prev_season_feature(
        season_end,
        ref_col=cols.post_tpoint_performance,
        new_col=cols.prev_season_tpoint_performance,
        required_cols=[cols.season, cols.div, cols.team, cols.post_tpoint_performance],
        fallback_function=_fallback_function,
        fillval=0,
    )

    return season_end


def _add_prev_season_feature(
    season_end: pd.DataFrame,
    *,
    ref_col: str,
    new_col: str,
    required_cols: list[str],
    fallback_function: Callable[[pd.DataFrame, pd.Series, str, str], pd.Series],
    fillval: int,
) -> pd.DataFrame:
    logger.info(f"Adding {new_col} to the DataFrame...")
    check_columns(season_end, required_cols)

    team_season = prev_season_value(season_end, new_col=new_col, ref_col=ref_col)
    first_season = season_end[cols.season].min()
    mask_no_prev_value = team_season["prev_season"] < first_season
    team_season.loc[mask_no_prev_value, new_col] = fallback_function(
        team_season, mask_no_prev_value, ref_col, new_col
    )
    team_season[new_col] = team_season[new_col].fillna(fillval)
    season_end = season_end.merge(
        team_season[[cols.season, cols.team, new_col]],
        on=merge_on,
        how="left",
    )

    return season_end
