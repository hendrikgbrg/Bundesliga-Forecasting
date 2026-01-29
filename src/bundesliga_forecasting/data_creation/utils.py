from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


def ensure_dir(paths: list[Path], dir_types: list[Literal["src", "target"]]) -> None:
    """
    Description:

    Usage location:
        data_creation/io/clean.py
        data_creation/io/merge.py
        data_creation/io/prepare.py

    Args:
        paths (list[Path]): _description_
        dir_types (list[Literal[&quot;src&quot;, &quot;target&quot;]]): _description_

    Raises:
        ValueError: _description_
        FileNotFoundError: _description_
    """
    if len(paths) != len(dir_types):
        raise ValueError(
            "The list objects 'paths' and 'dir_types' have to be of the same length."
        )
    for path, dir_type in zip(paths, dir_types):
        if not path.exists():
            if dir_type == "src":
                raise FileNotFoundError(f"Source directory does not exist: {path}")
            if dir_type == "target":
                path.mkdir(parents=True, exist_ok=True)
                logger.info("Created directory: %s", path)
        else:
            if not path.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {path}.")
            if dir_type == "src":
                if not any(path.iterdir()):
                    raise FileNotFoundError("Source directory is empty.")
                else:
                    logger.info(f"Reading file data from {path} ...")


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
