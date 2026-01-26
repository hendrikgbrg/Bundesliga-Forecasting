from pathlib import Path

import pandas as pd


def merge_files(
    src_dir: Path, target_dir: Path, *, col_names: list[str], sort_cols: list[str]
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
    target_dir.mkdir(parents=True, exist_ok=True)

    csv_files = [
        file
        for file in src_dir.iterdir()
        if file.is_file() and file.suffix.lower() == ".csv"
    ]

    df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)
    df.columns = col_names
    df.to_csv(target_dir / "data.csv", index=False)
