import logging
from pathlib import Path

import pandas as pd

from bundesliga_forecasting.data_creation import config, utils
from bundesliga_forecasting.data_creation.base import (
    add_season,
    sort_matches,
    team_match_split,
)

logger = logging.getLogger(__name__)

PATHS = config.PATHS


def prepare(
    src_dir: Path = PATHS.merged,
    target_dir: Path = PATHS.prepared,
    src_file: str = PATHS.m_filename,
    target_file: str = PATHS.p_filename,
) -> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        add_season (_type_): _description_
        team_match_split (_type_): _description_
    """
    logger.info("Starting file preparation...")

    utils.ensure_dir([src_dir, target_dir], ["src", "target"])

    input_path = src_dir / src_file
    output_path = target_dir / target_file

    df = pd.read_csv(input_path, parse_dates=[config.DATE_COL], dayfirst=True)
    df = add_season(df)
    df = team_match_split(df)
    df = sort_matches(df)

    df.to_csv(output_path, index=False)

    logger.info("File prepared successfully and saved to %s", output_path)
