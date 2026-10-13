
from pathlib import Path

# Location of the original dataset files
DATA_DIR = (
    Path(__file__).parent
    / "Data"
    / "Raw"
    / "unsw_nb15"
)

print("Dataset folder:", DATA_DIR)
print("\nChecking dataset files...\n")

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    print("ERROR: No CSV files found.")
    print("Check the folder path and file locations.")
else:
    for file in csv_files:
        size_mb = file.stat().st_size / (1024 * 1024)

        print(f"File: {file.name}")
        print(f"Size: {size_mb:.2f} MB")
        print("-" * 40)
