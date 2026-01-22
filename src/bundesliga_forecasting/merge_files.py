from pathlib import Path
import pandas as pd

base_dir = Path(__file__).resolve().parent
src_dir = (base_dir / "../../data/03_Cleaned").resolve()
target_dir = (base_dir / "../../data/04_Merged").resolve()
target_dir.mkdir(parents=True, exist_ok=True)

csv_files = [
    file
    for file in src_dir.iterdir()
    if file.is_file() and file.suffix.lower() == ".csv"
]

df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)

columns = ["Div", "Date", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"]

df.columns = columns

df.to_csv(target_dir / "data.csv", index=False)
