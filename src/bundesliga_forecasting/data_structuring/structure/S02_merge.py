import logging
from pathlib import Path

import pandas as pd

from bundesliga_forecasting.BL_config import COLUMNS, PATHS
from bundesliga_forecasting.BL_utils import ensure_dir, read_csv, save_to_csv
from bundesliga_forecasting.data_structuring.S_utils import detect_csv_files

logger = logging.getLogger(__name__)
paths = PATHS
cols = COLUMNS


def merge(
    src_dir: Path = paths.cleaned,
    target_dir: Path = paths.merged,
    target_file: str = paths.merged_file,
) -> None:
    """
    Description:
        1st - Concatinate all CSV-files onto each other in one data frame
        2nd - SSave the data frame to the target directory

    Usage location:
        data_creation/pipeline.py

    Args:
        src_dir (Path): _description_
        target_dir (Path): _description_
        col_names (list[str]): _description_
    """

    logger.info("Starting file merging...")

    ensure_dir([src_dir, target_dir], ["src", "target"])

    output_path = target_dir / target_file

    csv_files = detect_csv_files(src_dir)

    df = pd.concat((read_csv(file) for file in csv_files), ignore_index=True)

    save_to_csv(df, output_path)

    logger.info(
        f"{len(csv_files)} files merged successfully and saved in {output_path}."
    )
