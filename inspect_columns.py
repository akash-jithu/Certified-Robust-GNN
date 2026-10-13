
from pathlib import Path
import pandas as pd

DATA_DIR = (
    Path(__file__).parent
    / "Data"
    / "Raw"
    / "unsw_nb15"
)

# Inspect the first original CSV partition
csv_path = DATA_DIR / "UNSW-NB15_1.csv"

print("Reading:", csv_path.name)

# The original partitions may not contain column headers.
# First inspect the file exactly as stored.
sample = pd.read_csv(
    csv_path,
    header=None,
    nrows=5,
    low_memory=False
)

print("\nFirst 5 rows:")
print(sample.to_string(index=False))

print("\nNumber of columns:", len(sample.columns))
print("\nData types:")
print(sample.dtypes)
