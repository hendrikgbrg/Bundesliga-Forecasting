from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_csv_files(path: Path) -> list:
    """
    Description:

    Usage location:
        data_creation/io/clean.py
        data_creation/io/merge.py
        data_creation/io/prepare.py

    Args:
        path (Path): _description_

    Raises:
        ValueError: _description_

    Returns:
        list: _description_
    """
    csv_files = [
        file
        for file in path.iterdir()
        if file.is_file() and file.suffix.lower() == ".csv"
    ]
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV-files found in {path}.")
    return csv_files
