
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

previous_last_row = None

for i in range(1, 5):
    path = DATA_DIR / f"UNSW-NB15_{i}.csv"

    # Read the first record in this partition
    first_row = pd.read_csv(
        path,
        header=None,
        names=columns,
        nrows=1
    ).iloc[0]

    # Read the last record without loading the entire file
    last_chunk = None

    for chunk in pd.read_csv(
        path,
        header=None,
        names=columns,
        chunksize=100_000,
        low_memory=False
    ):
        last_chunk = chunk

    last_row = last_chunk.iloc[-1]

    print(f"\nPartition {i}: {path.name}")

    if previous_last_row is not None:
        same = previous_last_row.equals(first_row)

        print(
            "Previous partition's last row matches "
            "this partition's first row:",
            same
        )

        if same:
            print(
                "Possible overlapping boundary record detected."
            )
        else:
            print("No exact boundary match detected.")

    previous_last_row = last_row

print("\nBoundary check complete.")
