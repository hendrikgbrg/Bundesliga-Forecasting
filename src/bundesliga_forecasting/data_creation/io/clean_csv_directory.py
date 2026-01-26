from pathlib import Path

import pandas as pd

from bundesliga_forecasting.data_creation.base import adjust_team_names, extract_columns


def clean_csv_directory(
    src_dir: Path,
    target_dir: Path,
    *,
    col_names: list[str],
    rename_map: dict[str, str],
    encoding: str = "latin1",
) -> None:
    """
    Description:
        1st - Read each CSV-file in the source directory as a data frame
        2nd - Restrict the data frame to necessary columns
        3rd - Remove empty rows and spaces
        4th - Adjust team names
        5th - Save the data frame to the target directory

    Usage location:
        data_creation/pipeline.py

    Args:
        src_dir (Path): _description_
        target_dir (Path): _description_
        col_names (list[str]): _description_
        rename_map (dict[str, str]): _description_
        encoding (str, optional): _description_. Defaults to "latin1".
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    csv_files = [
        file
        for file in src_dir.iterdir()
        if file.is_file() and file.suffix.lower() == ".csv"
    ]
    for file in csv_files:
        print(f"Processing: {file.name}")

        # read file line by line
        with open(file, encoding=encoding) as f:
            lines = f.readlines()
            rows = extract_columns(lines, col_names)
            df = pd.DataFrame(data=rows, columns=col_names)
            df = adjust_team_names(df, col_names=col_names, rename_map=rename_map)

            df.to_csv(target_dir / file.name, index=False)
            print(f"Saved: {file.name}, shape: {df.shape}")
