
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).parent
DATA_FILE = (
    PROJECT_DIR
    / "Data"
    / "Processed"
    / "unsw_nb15_combined.csv"
)

print("Checking combined dataset...\n")

# Load only a small sample first
sample = pd.read_csv(DATA_FILE, nrows=5)

print("First 5 rows:")
print(sample.head())

print("\nDataset columns:")
print(sample.columns.tolist())

# Count rows and labels in chunks
total_rows = 0
label_counts = {0: 0, 1: 0}
invalid_labels = 0

for chunk in pd.read_csv(DATA_FILE, chunksize=100_000):
    total_rows += len(chunk)

    counts = chunk["Label"].value_counts()
    label_counts[0] += int(counts.get(0, 0))
    label_counts[1] += int(counts.get(1, 0))

    invalid_labels += int(chunk["Label"].isna().sum())
    invalid_labels += int((~chunk["Label"].isin([0, 1])).sum())

print("\nValidation results")
print("------------------")
print(f"Total records: {total_rows:,}")
print(f"Total columns: {len(sample.columns)}")
print(f"Normal records (0): {label_counts[0]:,}")
print(f"Attack records (1): {label_counts[1]:,}")
print(f"Invalid labels: {invalid_labels}")

print("\nValidation complete.")
