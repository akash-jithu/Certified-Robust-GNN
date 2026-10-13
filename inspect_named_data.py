
from pathlib import Path
import pandas as pd

DATA_DIR = (
    Path(__file__).parent
    / "Data"
    / "Raw"
    / "unsw_nb15"
)

# Load the feature description file
feature_path = next(DATA_DIR.glob("*features*.csv"))

features = pd.read_csv(
    feature_path,
    encoding="cp1252",
    skipinitialspace=True
)

# Extract the 49 feature names
column_names = (
    features["Name"]
    .astype(str)
    .str.strip()
    .tolist()
)

if len(column_names) != 49:
    raise ValueError(
        f"Expected 49 column names, found {len(column_names)}"
    )

print("Number of column names:", len(column_names))

# Read only 10 rows from the first partition
csv_path = DATA_DIR / "UNSW-NB15_1.csv"

sample = pd.read_csv(
    csv_path,
    header=None,
    names=column_names,
    nrows=10,
    low_memory=False
)

print("\nDataset columns:")
print(sample.columns.tolist())

# Display the columns most useful for our project
important_columns = [
    "srcip",
    "sport",
    "dstip",
    "dsport",
    "proto",
    "dur",
    "attack_cat",
    "Label"
]

print("\nImportant columns from the first 10 records:")
print(sample[important_columns].to_string(index=False))

print("\nData types:")
print(sample[important_columns].dtypes)

print("\nMissing values in this sample:")
print(sample[important_columns].isnull().sum())
