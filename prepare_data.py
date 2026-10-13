
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).parent

DATA_FILE = (
    PROJECT_DIR
    / "Data"
    / "Processed"
    / "unsw_nb15_combined.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "Data"
    / "Processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = OUTPUT_DIR / "gnn_sample_data.npz"
PREPROCESSOR_FILE = OUTPUT_DIR / "gnn_preprocessor.joblib"


# ==================================================
# 2. CHECK DATASET
# ==================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}"
    )

print("=" * 55)
print("CERTIFIED ROBUST GNN - DATA PREPROCESSING")
print("=" * 55)

print("\nSampling flows across the dataset...")


# ==================================================
# 3. LOAD A REPRESENTATIVE SAMPLE
# ==================================================

samples = []

for i, chunk in enumerate(
    pd.read_csv(
        DATA_FILE,
        chunksize=100_000,
        low_memory=False
    )
):
    sample_size = min(
        2_000,
        len(chunk)
    )

    chunk_sample = chunk.sample(
        n=sample_size,
        random_state=42 + i
    )

    samples.append(chunk_sample)

    print(
        f"Processed chunk {i + 1}: "
        f"sampled {sample_size:,} rows"
    )

df = pd.concat(
    samples,
    ignore_index=True
)

# Limit the development sample to 50,000 flows
if len(df) > 50_000:
    df = df.sample(
        n=50_000,
        random_state=42
    ).reset_index(drop=True)

# Normalize column names
df.columns = [
    col.strip().replace("__", "_")
    for col in df.columns
]

print(f"\nSample loaded: {len(df):,} flows")


# ==================================================
# 4. CLEAN THE TARGET LABEL
# ==================================================

df = df.dropna(
    subset=["Label"]
).copy()

df["Label"] = pd.to_numeric(
    df["Label"],
    errors="coerce"
)

df = df.dropna(
    subset=["Label"]
).copy()

df["Label"] = df["Label"].astype(int)

# Keep only normal (0) and attack (1) records
df = df[
    df["Label"].isin([0, 1])
].copy()

# Ensure source and destination addresses are available
df = df.dropna(
    subset=["srcip", "dstip"]
).copy()

# Reset indices to keep all arrays aligned
df = df.reset_index(drop=True)

y = df["Label"].copy()


# ==================================================
# 5. EXTRACT GRAPH IDENTIFIERS
# ==================================================

# IP addresses identify graph nodes.
# They are saved separately and not used as numeric features.

srcip = (
    df["srcip"]
    .astype(str)
    .str.strip()
    .to_numpy()
)

dstip = (
    df["dstip"]
    .astype(str)
    .str.strip()
    .to_numpy()
)


# ==================================================
# 6. SELECT INPUT FEATURES
# ==================================================

excluded_columns = [
    "srcip",
    "dstip",
    "Stime",
    "Ltime",
    "attack_cat",
    "Label"
]

X = df.drop(
    columns=excluded_columns,
    errors="ignore"
).copy()

# Categorical columns with manageable cardinality
categorical_columns = [
    col
    for col in ["proto", "state", "service"]
    if col in X.columns
]

# Treat all remaining columns as numeric features
numeric_columns = [
    col
    for col in X.columns
    if col not in categorical_columns
]

print(
    f"\nInput features before encoding: "
    f"{X.shape[1]}"
)

print(
    f"Numeric columns: {len(numeric_columns)}"
)

print(
    f"Categorical columns: {len(categorical_columns)}"
)


# ==================================================
# 7. CLEAN CATEGORICAL FEATURES
# ==================================================

for col in categorical_columns:
    X[col] = (
        X[col]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
        .fillna("unknown")
        .astype(str)
    )


# ==================================================
# 8. CLEAN NUMERIC FEATURES
# ==================================================

for col in numeric_columns:
    X[col] = pd.to_numeric(
        X[col]
        .astype("string")
        .str.strip(),
        errors="coerce"
    )

# Replace infinite values with missing values
X[numeric_columns] = X[numeric_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

# Remove entirely empty numeric columns, if any
empty_numeric_columns = [
    col
    for col in numeric_columns
    if X[col].isna().all()
]

if empty_numeric_columns:
    print(
        "\nRemoving entirely empty numeric columns:",
        empty_numeric_columns
    )

    X = X.drop(
        columns=empty_numeric_columns
    )

    numeric_columns = [
        col
        for col in numeric_columns
        if col not in empty_numeric_columns
    ]


# ==================================================
# 9. SPLIT FEATURES, LABELS AND IP ADDRESSES
# ==================================================

# Keep the same row indices across every array.
all_indices = np.arange(len(X))

train_indices, temp_indices = train_test_split(
    all_indices,
    test_size=0.30,
    random_state=42,
    stratify=y
)

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=42,
    stratify=y.iloc[temp_indices]
)

X_train = X.iloc[train_indices]
X_val = X.iloc[val_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_val = y.iloc[val_indices]
y_test = y.iloc[test_indices]

srcip_train = srcip[train_indices]
dstip_train = dstip[train_indices]

srcip_val = srcip[val_indices]
dstip_val = dstip[val_indices]

srcip_test = srcip[test_indices]
dstip_test = dstip[test_indices]


# ==================================================
# 10. BUILD PREPROCESSING PIPELINES
# ==================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),
    (
        "scaler",
        StandardScaler()
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])

transformers = []

if numeric_columns:
    transformers.append(
        (
            "numeric",
            numeric_pipeline,
            numeric_columns
        )
    )

if categorical_columns:
    transformers.append(
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    )

preprocessor = ColumnTransformer(
    transformers=transformers
)


# ==================================================
# 11. FIT AND TRANSFORM THE DATA
# ==================================================

print("\nPreprocessing features...")

# Fit using training data only to avoid preprocessing leakage
X_train_processed = preprocessor.fit_transform(
    X_train
)

X_val_processed = preprocessor.transform(
    X_val
)

X_test_processed = preprocessor.transform(
    X_test
)

# Convert sparse matrices to dense arrays
if issparse(X_train_processed):
    X_train_processed = X_train_processed.toarray()

if issparse(X_val_processed):
    X_val_processed = X_val_processed.toarray()

if issparse(X_test_processed):
    X_test_processed = X_test_processed.toarray()

# Use float32 to reduce memory usage
X_train_processed = np.asarray(
    X_train_processed,
    dtype=np.float32
)

X_val_processed = np.asarray(
    X_val_processed,
    dtype=np.float32
)

X_test_processed = np.asarray(
    X_test_processed,
    dtype=np.float32
)


# ==================================================
# 12. SAVE PROCESSED DATA
# ==================================================

np.savez_compressed(
    OUTPUT_FILE,

    # Processed input features
    X_train=X_train_processed,
    X_val=X_val_processed,
    X_test=X_test_processed,

    # Binary labels
    y_train=y_train.to_numpy(dtype=np.int64),
    y_val=y_val.to_numpy(dtype=np.int64),
    y_test=y_test.to_numpy(dtype=np.int64),

    # Source and destination IP addresses
    srcip_train=srcip_train,
    dstip_train=dstip_train,

    srcip_val=srcip_val,
    dstip_val=dstip_val,

    srcip_test=srcip_test,
    dstip_test=dstip_test
)

# Save the fitted preprocessing pipeline
joblib.dump(
    preprocessor,
    PREPROCESSOR_FILE
)


# ==================================================
# 13. DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 55)
print("PREPROCESSING RESULTS")
print("=" * 55)

print(f"Total sampled flows: {len(df):,}")
print(f"Original dataset columns: {len(df.columns)}")
print(f"Input features before encoding: {X.shape[1]}")

print("\nDataset splits:")
print(f"Training samples:   {len(y_train):,}")
print(f"Validation samples: {len(y_val):,}")
print(f"Testing samples:    {len(y_test):,}")

print("\nProcessed feature shapes:")
print(f"Training:   {X_train_processed.shape}")
print(f"Validation: {X_val_processed.shape}")
print(f"Testing:    {X_test_processed.shape}")

print("\nLabel distribution in the sample:")
print(
    y.value_counts()
    .sort_index()
    .rename(index={
        0: "Normal (0)",
        1: "Attack (1)"
    })
)

print("\nAttack rates:")
print(f"Training:   {y_train.mean() * 100:.2f}%")
print(f"Validation: {y_val.mean() * 100:.2f}%")
print(f"Testing:    {y_test.mean() * 100:.2f}%")

print("\nSaved files:")
print(f"Processed data: {OUTPUT_FILE}")
print(f"Preprocessor:   {PREPROCESSOR_FILE}")

print("\nSUCCESS: Data preprocessing and saving completed.")
