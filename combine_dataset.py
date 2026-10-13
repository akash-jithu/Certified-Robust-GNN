
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "Data" / "Raw" / "unsw_nb15"
OUTPUT_DIR = PROJECT_DIR / "Data" / "Processed"
OUTPUT_FILE = OUTPUT_DIR / "unsw_nb15_combined.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read the feature names
feature_path = next(DATA_DIR.glob("*features*.csv"))

features = pd.read_csv(
    feature_path,
    encoding="cp1252",
    skipinitialspace=True
)

columns = (
    features["Name"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", "_", regex=True)
    .tolist()
)

total_rows = 0
label_counts = {0: 0, 1: 0}

# Combine the four CSV partitions
for i in range(1, 5):
    file_path = DATA_DIR / f"UNSW-NB15_{i}.csv"

    print(f"\nProcessing {file_path.name}...")

    # Skip the first record in partitions 2, 3, and 4.
    # These records were verified as exact boundary overlaps.
    skip_rows = 1 if i > 1 else 0

    first_chunk = True

    for chunk in pd.read_csv(
        file_path,
        header=None,
        names=columns,
        skiprows=skip_rows,
        chunksize=100_000,
        low_memory=False
    ):
        # Count labels before writing
        counts = chunk["Label"].value_counts()

        label_counts[0] += int(counts.get(0, 0))
        label_counts[1] += int(counts.get(1, 0))

        # Write each chunk without loading the entire dataset into RAM
        chunk.to_csv(
            OUTPUT_FILE,
            index=False,
            mode="w" if first_chunk and i == 1 else "a",
            header=first_chunk and i == 1
        )

        total_rows += len(chunk)
        first_chunk = False

    print(f"Finished partition {i}")

print("\nDataset combination complete!")
print(f"Output file: {OUTPUT_FILE}")
print(f"Total rows: {total_rows:,}")
print(f"Normal records (0): {label_counts[0]:,}")
print(f"Attack records (1): {label_counts[1]:,}")
print(f"Total columns: {len(columns)}")
print(f"Output size: {OUTPUT_FILE.stat().st_size / (1024**2):.2f} MB")
