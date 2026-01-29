import logging
from pathlib import Path

import pandas as pd

from bundesliga_forecasting.data_creation import config, utils
from bundesliga_forecasting.data_creation.base import (
    adjust_team_names,
    extract_columns,
)

logger = logging.getLogger(__name__)

PATHS = config.PATHS
raw_cols = config.COLUMNS.raw


def clean(
    src_dir: Path = PATHS.raw,
    target_dir: Path = PATHS.cleaned,
    *,
    encoding: str = config.CSV_ENCODING,
) -> None:
    """
    Description:
        Step 1 -> Read the lines of each CSV-file in the source directory
        Step 2 -> Create the data frame with required columns & remove empty rows and spaces
        Step 3 -> Adjust team names
        Step 4 -> Save the data frame to the target directory

    Usage location:
        data_creation/pipeline.py

    Args:
        src_dir (Path): _description_
        target_dir (Path): _description_
        col_names (list[str]): _description_
        rename_map (dict[str, str]): _description_
        encoding (str, optional): _description_. Defaults to "latin1".
    """

    logger.info("Starting file cleaning...")

    utils.ensure_dir([src_dir, target_dir], ["src", "target"])

    csv_files = utils.detect_csv_files(src_dir)

    for file in csv_files:
        logger.info("Processing: %s", file.name)
        with open(file, encoding=encoding) as f:
            lines = f.readlines()

            # Step 2:
            rows = extract_columns(lines)
            df = pd.DataFrame(data=rows, columns=raw_cols)

            # Step 3:
            df = adjust_team_names(df)

            # Step 4:
            df.to_csv(target_dir / file.name, index=False)

    logger.info(
        f"{len(csv_files)} files cleaned successfully and saved in {target_dir}."
    )
