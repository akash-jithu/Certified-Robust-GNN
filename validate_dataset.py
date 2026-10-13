
from pathlib import Path
import pandas as pd

DATA_DIR = (
    Path(__file__).parent
    / "Data"
    / "Raw"
    / "unsw_nb15"
)

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

all_ips = set()

for i in range(1, 5):
    path = DATA_DIR / f"UNSW-NB15_{i}.csv"

    print("\n" + "=" * 55)
    print("FILE:", path.name)

    # Inspect the beginning and end of each partition
    first = pd.read_csv(
        path,
        header=None,
        names=columns,
        nrows=2
    )

    last = None
    rows = 0
    bad_labels = 0

    for chunk in pd.read_csv(
        path,
        header=None,
        names=columns,
        chunksize=100_000,
        low_memory=False
    ):
        rows += len(chunk)
        last = chunk.tail(2)

        labels = pd.to_numeric(
            chunk["Label"],
            errors="coerce"
        )
        bad_labels += int(labels.isna().sum())

        all_ips.update(chunk["srcip"].dropna().unique())
        all_ips.update(chunk["dstip"].dropna().unique())

    print("Rows:", f"{rows:,}")
    print("Invalid/missing binary labels:", bad_labels)

    print("\nFirst two source/destination pairs:")
    print(first[["srcip", "dstip", "Label"]].to_string(index=False))

    print("\nLast two source/destination pairs:")
    print(last[["srcip", "dstip", "Label"]].to_string(index=False))

print("\n" + "=" * 55)
print("GRAPH STRUCTURE CHECK")

print("Unique IP addresses:", len(all_ips))

print("\nIP addresses:")
for ip in sorted(all_ips, key=str):
    print(ip)

print("\nValidation finished.")
