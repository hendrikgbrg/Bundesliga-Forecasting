import logging
from pathlib import Path

import pandas as pd

from bundesliga_forecasting.data_creation import config, utils

logger = logging.getLogger(__name__)

PATHS = config.PATHS


def merge(
    src_dir: Path = PATHS.cleaned,
    target_dir: Path = PATHS.merged,
    target_file: str = PATHS.m_filename,
) -> None:
    """
    Description:
        1st - Concatinate all CSV-files onto each other in one data frame
        2nd - Sort the data frame
        3rd - Save the data frame to the target directory

    Usage location:
        data_creation/pipeline.py

    Args:
        src_dir (Path): _description_
        target_dir (Path): _description_
        col_names (list[str]): _description_
        sort_cols (list[str]): _description_
    """

    logger.info("Starting file merging...")

    utils.ensure_dir([src_dir, target_dir], ["src", "target"])

    output_path = target_dir / target_file

    csv_files = utils.detect_csv_files(src_dir)

    df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)
    # df = sort_matches(df)
    df.to_csv(output_path, index=False)

    logger.info(
        f"{len(csv_files)} files merged successfully and saved in {output_path}."
    )
