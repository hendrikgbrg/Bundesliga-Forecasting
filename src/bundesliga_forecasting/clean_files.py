from pathlib import Path
import pandas as pd

base_dir = Path(__file__).resolve().parent
src_dir = (base_dir / "../../data/02_Renamed").resolve()
target_dir = (base_dir / "../../data/03_Cleaned").resolve()
target_dir.mkdir(parents=True, exist_ok=True)

columns = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]

csv_files = [
    file
    for file in src_dir.iterdir()
    if file.is_file() and file.suffix.lower() == ".csv"
]

for file in csv_files:
    print(f"Processing: {file.name}")
    cleaned_rows = []

    # read file line by line
    with open(file, encoding="latin1") as f:
        for line in f:
            fields = line.strip().split(",")

            if fields and fields[0] == "Div":
                col_indices = [fields.index(col) for col in columns]
                continue

            if any(field.strip() for field in fields):
                selected = [fields[idx] for idx in col_indices]
            else:
                continue

            cleaned_rows.append(selected)

    df = pd.DataFrame(cleaned_rows, columns=columns)
    df[["HomeTeam", "AwayTeam"]] = df[["HomeTeam", "AwayTeam"]].apply(
        lambda col: col.str.strip()
    )
    rename_map = {
        "Dusseldorf": "Fortuna Dusseldorf",
        "Leipzig": "VfB Leipzig",
        "F Koln": "Fortuna Koln",
    }
    df[["HomeTeam", "AwayTeam"]] = df[["HomeTeam", "AwayTeam"]].replace(rename_map)

    df.to_csv(target_dir / file.name, index=False)
    print(f"Saved: {file.name}, shape: {df.shape}")
