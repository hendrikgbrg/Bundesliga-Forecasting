from pathlib import Path
import shutil

base_dir = Path(__file__).resolve().parent
src_dir = (base_dir / "../../data/01_Raw").resolve()
target_dir = (base_dir / "../../data/02_Renamed").resolve()
target_dir.mkdir(parents=True, exist_ok=True)

csv_files = [
    file
    for file in src_dir.iterdir()
    if file.is_file() and file.suffix.lower() == ".csv"
]

for file in csv_files:
    prefix = "19" if file.name.startswith("9") else "20"
    file_name = prefix + file.name
    file_path = target_dir / file_name

    shutil.copy2(file, file_path)
