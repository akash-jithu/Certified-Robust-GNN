
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).parent
DATA_FILE = (
    PROJECT_DIR
    / "Data"
    / "Processed"
    / "unsw_nb15_combined.csv"
)

# Read the column names and a representative sample
sample = pd.read_csv(DATA_FILE, nrows=20_000, low_memory=False)

# Fix the inconsistent column name
sample.columns = [
    name.replace("__", "_") for name in sample.columns
]

print("DATASET SHAPE (sample)")
print("----------------------")
print(f"Sample rows: {len(sample):,}")
print(f"Columns: {len(sample.columns)}")

print("\nCOLUMN DATA TYPES")
print("-----------------")
print(sample.dtypes.to_string())

print("\nMISSING VALUES IN SAMPLE")
print("------------------------")
missing = sample.isna().sum()
print(missing[missing > 0].to_string())

print("\nCATEGORICAL COLUMNS")
print("-------------------")
categorical = sample.select_dtypes(
    include=["object", "category"]
).columns.tolist()
print(categorical)

print("\nNUMERIC COLUMNS")
print("---------------")
numeric = sample.select_dtypes(
    include=["number"]
).columns.tolist()
print(numeric)

print("\nPREPROCESSING INSPECTION COMPLETE")
