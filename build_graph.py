
from pathlib import Path

import numpy as np
import torch


# ==================================================
# 1. PATHS
# ==================================================

PROJECT_DIR = Path(__file__).parent

DATA_FILE = (
    PROJECT_DIR
    / "Data"
    / "Processed"
    / "gnn_sample_data.npz"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "Data"
    / "Processed"
    / "gnn_graph_data.pt"
)


# ==================================================
# 2. CHECK INPUT FILE
# ==================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Processed dataset not found:\n{DATA_FILE}\n"
        "Run prepare_data.py first."
    )

print("=" * 55)
print("CERTIFIED ROBUST GNN - GRAPH CONSTRUCTION")
print("=" * 55)


# ==================================================
# 3. LOAD PREPROCESSED DATA
# ==================================================

data = np.load(
    DATA_FILE,
    allow_pickle=True
)

X_train = data["X_train"]
X_val = data["X_val"]
X_test = data["X_test"]

y_train = data["y_train"]
y_val = data["y_val"]
y_test = data["y_test"]

srcip_train = data["srcip_train"].astype(str)
dstip_train = data["dstip_train"].astype(str)

srcip_val = data["srcip_val"].astype(str)
dstip_val = data["dstip_val"].astype(str)

srcip_test = data["srcip_test"].astype(str)
dstip_test = data["dstip_test"].astype(str)


# ==================================================
# 4. CREATE A CONSISTENT HOST-TO-NODE MAPPING
# ==================================================

all_ips = np.concatenate([
    srcip_train,
    dstip_train,
    srcip_val,
    dstip_val,
    srcip_test,
    dstip_test
])

unique_ips = np.unique(all_ips)

ip_to_node = {
    ip: index
    for index, ip in enumerate(unique_ips)
}

num_nodes = len(unique_ips)

print(f"\nUnique network hosts: {num_nodes}")


# ==================================================
# 5. CONVERT IP ADDRESSES TO NODE INDICES
# ==================================================

def make_edge_index(src_ips, dst_ips):
    """
    Convert source and destination IP addresses
    into PyTorch Geometric edge-index format.

    Shape: [2, number_of_flows]
    """
    src_nodes = np.array(
        [ip_to_node[ip] for ip in src_ips],
        dtype=np.int64
    )

    dst_nodes = np.array(
        [ip_to_node[ip] for ip in dst_ips],
        dtype=np.int64
    )

    return torch.tensor(
        np.stack([src_nodes, dst_nodes]),
        dtype=torch.long
    )


train_edge_index = make_edge_index(
    srcip_train,
    dstip_train
)

val_edge_index = make_edge_index(
    srcip_val,
    dstip_val
)

test_edge_index = make_edge_index(
    srcip_test,
    dstip_test
)


# ==================================================
# 6. CONVERT FEATURES AND LABELS TO TENSORS
# ==================================================

train_edge_features = torch.tensor(
    X_train,
    dtype=torch.float32
)

val_edge_features = torch.tensor(
    X_val,
    dtype=torch.float32
)

test_edge_features = torch.tensor(
    X_test,
    dtype=torch.float32
)

train_edge_labels = torch.tensor(
    y_train,
    dtype=torch.long
)

val_edge_labels = torch.tensor(
    y_val,
    dtype=torch.long
)

test_edge_labels = torch.tensor(
    y_test,
    dtype=torch.long
)


# ==================================================
# 7. CHECK GRAPH DATA
# ==================================================

assert train_edge_index.shape[1] == len(y_train)
assert val_edge_index.shape[1] == len(y_val)
assert test_edge_index.shape[1] == len(y_test)

assert train_edge_features.shape[0] == len(y_train)
assert val_edge_features.shape[0] == len(y_val)
assert test_edge_features.shape[0] == len(y_test)

assert train_edge_index.min().item() >= 0
assert test_edge_index.max().item() < num_nodes

print("\nGRAPH DATA SUMMARY")
print("-" * 40)

print(f"Number of nodes: {num_nodes}")

print(
    f"Training edges: {train_edge_index.shape[1]:,}"
)

print(
    f"Validation edges: {val_edge_index.shape[1]:,}"
)

print(
    f"Testing edges: {test_edge_index.shape[1]:,}"
)

print(
    f"Features per edge: {train_edge_features.shape[1]}"
)

print(
    f"Training labels: {train_edge_labels.shape}"
)

print(
    f"Validation labels: {val_edge_labels.shape}"
)

print(
    f"Testing labels: {test_edge_labels.shape}"
)


# ==================================================
# 8. SAVE GRAPH DATA
# ==================================================

graph_data = {
    "num_nodes": num_nodes,
    "unique_ips": unique_ips.tolist(),

    "train_edge_index": train_edge_index,
    "train_edge_features": train_edge_features,
    "train_edge_labels": train_edge_labels,

    "val_edge_index": val_edge_index,
    "val_edge_features": val_edge_features,
    "val_edge_labels": val_edge_labels,

    "test_edge_index": test_edge_index,
    "test_edge_features": test_edge_features,
    "test_edge_labels": test_edge_labels,
}

torch.save(
    graph_data,
    OUTPUT_FILE
)

print(f"\nGraph data saved to:\n{OUTPUT_FILE}")
print("\nSUCCESS: Graph construction completed.")
