
from pathlib import Path
from collections import Counter
import pandas as pd

DATA_DIR = (
    Path(__file__).parent
    / "Data"
    / "Raw"
    / "unsw_nb15"
)

CHUNK_SIZE = 100_000

# Read feature names and clean accidental whitespace
feature_path = next(DATA_DIR.glob("*features*.csv"))

features = pd.read_csv(
    feature_path,
    encoding="cp1252",
    skipinitialspace=True
)

column_names = (
    features["Name"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", "_", regex=True)
    .tolist()
)

label_counts = Counter()
attack_counts = Counter()
missing_counts = Counter()
unique_ips = set()

total_rows = 0
partition_rows = Counter()

csv_files = [
    DATA_DIR / f"UNSW-NB15_{i}.csv"
    for i in range(1, 5)
]

for csv_path in csv_files:
    print(f"\nAnalyzing: {csv_path.name}")
    file_rows = 0

    for chunk in pd.read_csv(
        csv_path,
        header=None,
        names=column_names,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):
        rows = len(chunk)
        file_rows += rows
        total_rows += rows

        # Count normal and attack labels
        labels = chunk["Label"].value_counts(dropna=False)
        label_counts.update(labels.to_dict())

        # Count attack categories
        categories = (
            chunk["attack_cat"]
            .astype("string")
            .str.strip()
        )

        # Missing categories on normal flows are expected.
        normal = chunk["Label"].eq(0)
        categories = categories.mask(
            normal & categories.isna(),
            "Normal"
        )

        # Keep unexpected missing attack categories visible.
        categories = categories.fillna(
            "Missing attack category"
        )

        attack_counts.update(categories.tolist())

        # Count missing values for each column
        missing_counts.update(
            chunk.isna().sum().to_dict()
        )

        # Collect IP addresses for unique-node estimate
        unique_ips.update(chunk["srcip"].dropna().unique())
        unique_ips.update(chunk["dstip"].dropna().unique())

    partition_rows[csv_path.name] = file_rows
    print(f"Rows processed: {file_rows:,}")

print("\n" + "=" * 50)
print("DATASET SUMMARY")
print("=" * 50)

print(f"\nTotal flow records: {total_rows:,}")

print("\nRecords per partition:")
for name, count in partition_rows.items():
    print(f"{name}: {count:,}")

print("\nBinary label distribution:")
for label, count in sorted(label_counts.items(), key=lambda x: str(x[0])):
    meaning = {
        0: "Normal",
        1: "Attack"
    }.get(label, "Other or missing")
    print(f"{label} ({meaning}): {count:,}")

print("\nAttack category distribution:")
for category, count in attack_counts.most_common():
    print(f"{category}: {count:,}")

print("\nUnique IP addresses:", f"{len(unique_ips):,}")

print("\nMissing values in important columns:")
for col in [
    "srcip", "dstip", "proto",
    "dur", "attack_cat", "Label"
]:
    print(f"{col}: {missing_counts[col]:,}")
