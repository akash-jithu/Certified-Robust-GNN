
from pathlib import Path
import pandas as pd

DATA_DIR = (
    Path(__file__).parent
    / "Data"
    / "Raw"
    / "unsw_nb15"
)

feature_files = list(DATA_DIR.glob("*features*.csv"))

if not feature_files:
    raise FileNotFoundError(
        "Feature description CSV was not found."
    )

feature_path = feature_files[0]

print("Feature file:", feature_path.name)

# Try Windows-compatible encoding
features = pd.read_csv(
    feature_path,
    encoding="cp1252",
    skipinitialspace=True
)

print("\nColumn names:")
print(features.columns.tolist())

print("\nFirst 10 feature descriptions:")
print(features.head(10).to_string(index=False))

print("\nLast 5 feature descriptions:")
print(features.tail(5).to_string(index=False))

print("\nTotal rows in feature file:", len(features))
