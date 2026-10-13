
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).parent

DATA_FILE = (
    PROJECT_DIR
    / "Data"
    / "Processed"
    / "gnn_graph_data.pt"
)

MODEL_FILE = (
    PROJECT_DIR
    / "Data"
    / "Processed"
    / "baseline_gcn_model.pt"
)

SEED = 42
HIDDEN_DIM = 64
EPOCHS = 30
LEARNING_RATE = 0.001

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cpu")

print("=" * 55)
print("BASELINE GCN - NETWORK INTRUSION DETECTION")
print("=" * 55)
print("Device:", DEVICE)


# ==================================================
# 2. LOAD GRAPH DATA
# ==================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Graph data not found: {DATA_FILE}\n"
        "Run build_graph.py first."
    )

# Only load the trusted file created by build_graph.py.
try:
    graph = torch.load(
        DATA_FILE,
        map_location=DEVICE,
        weights_only=False,
    )
except TypeError:
    graph = torch.load(
        DATA_FILE,
        map_location=DEVICE,
    )

num_nodes = graph["num_nodes"]

train_edges = graph["train_edge_index"].to(DEVICE)
val_edges = graph["val_edge_index"].to(DEVICE)
test_edges = graph["test_edge_index"].to(DEVICE)

X_train = graph["train_edge_features"].to(DEVICE)
X_val = graph["val_edge_features"].to(DEVICE)
X_test = graph["test_edge_features"].to(DEVICE)

y_train = graph["train_edge_labels"].to(DEVICE)
y_val = graph["val_edge_labels"].to(DEVICE)
y_test = graph["test_edge_labels"].to(DEVICE)

print("\nLoaded data:")
print("Nodes:", num_nodes)
print("Training flows:", len(y_train))
print("Validation flows:", len(y_val))
print("Test flows:", len(y_test))
print("Flow feature dimension:", X_train.shape[1])


# ==================================================
# 3. BUILD NODE FEATURES
# ==================================================

# Each host receives the average of its incident
# TRAINING flow features. Validation and test flows
# are not used to construct node features.

feature_dim = X_train.shape[1]

node_features = torch.zeros(
    (num_nodes, feature_dim),
    dtype=torch.float32,
    device=DEVICE,
)

node_counts = torch.zeros(
    num_nodes,
    dtype=torch.float32,
    device=DEVICE,
)

src = train_edges[0]
dst = train_edges[1]

# Aggregate source-flow features
node_features.index_add_(
    0,
    src,
    X_train,
)

node_counts.index_add_(
    0,
    src,
    torch.ones(
        len(src),
        dtype=torch.float32,
        device=DEVICE,
    ),
)

# Aggregate destination-flow features
node_features.index_add_(
    0,
    dst,
    X_train,
)

node_counts.index_add_(
    0,
    dst,
    torch.ones(
        len(dst),
        dtype=torch.float32,
        device=DEVICE,
    ),
)

node_features = node_features / node_counts.clamp(min=1).unsqueeze(1)

print("\nNode feature shape:", tuple(node_features.shape))


# ==================================================
# 4. CREATE THE TRAINING MESSAGE-PASSING GRAPH
# ==================================================

# Add reverse connections so information can flow
# in both directions during message passing.
# This does not add validation or test flows.

reverse_train_edges = train_edges.flip(0)

message_edges = torch.cat(
    [train_edges, reverse_train_edges],
    dim=1,
)

# Remove duplicate connections for the GCN graph.
# Repeated flows remain separate examples for the
# edge classifier.
message_edges = torch.unique(
    message_edges,
    dim=1,
)

print(
    "Unique directed message-passing connections:",
    message_edges.shape[1],
)


# ==================================================
# 5. DEFINE THE GCN + EDGE CLASSIFIER
# ==================================================

class GCNFlowClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        flow_dim,
    ):
        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            hidden_dim,
        )

        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim,
        )

        # Input:
        # source embedding
        # destination embedding
        # original flow features
        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_dim * 2 + flow_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(
                hidden_dim,
                2,
            ),
        )

    def encode(
        self,
        node_x,
        edge_index,
    ):
        x = self.conv1(
            node_x,
            edge_index,
        )

        x = F.relu(x)
        x = F.dropout(
            x,
            p=0.2,
            training=self.training,
        )

        x = self.conv2(
            x,
            edge_index,
        )

        return F.relu(x)

    def predict_flows(
        self,
        node_embeddings,
        flow_edges,
        flow_features,
    ):
        source_embeddings = node_embeddings[
            flow_edges[0]
        ]

        destination_embeddings = node_embeddings[
            flow_edges[1]
        ]

        combined = torch.cat(
            [
                source_embeddings,
                destination_embeddings,
                flow_features,
            ],
            dim=1,
        )

        return self.classifier(combined)

    def forward(
        self,
        node_x,
        message_edge_index,
        flow_edges,
        flow_features,
    ):
        embeddings = self.encode(
            node_x,
            message_edge_index,
        )

        return self.predict_flows(
            embeddings,
            flow_edges,
            flow_features,
        )


model = GCNFlowClassifier(
    input_dim=feature_dim,
    hidden_dim=HIDDEN_DIM,
    flow_dim=feature_dim,
).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=5e-4,
)

criterion = nn.CrossEntropyLoss()

print("\nModel:")
print(model)


# ==================================================
# 6. EVALUATION FUNCTION
# ==================================================

def evaluate(
    features,
    labels,
    edges,
):
    model.eval()

    with torch.no_grad():
        logits = model(
            node_features,
            message_edges,
            edges,
            features,
        )

        loss = criterion(
            logits,
            labels,
        ).item()

        predictions = logits.argmax(dim=1)

    y_true = labels.cpu().numpy()
    y_pred = predictions.cpu().numpy()

    metrics = {
        "loss": loss,
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }

    return metrics, y_true, y_pred


# ==================================================
# 7. TRAIN THE BASELINE
# ==================================================

best_val_f1 = -1.0

print("\nStarting training...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    train_logits = model(
        node_features,
        message_edges,
        train_edges,
        X_train,
    )

    train_loss = criterion(
        train_logits,
        y_train,
    )

    train_loss.backward()
    optimizer.step()

    val_metrics, _, _ = evaluate(
        X_val,
        y_val,
        val_edges,
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss.item():.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val Accuracy: {val_metrics['accuracy']:.4f} | "
        f"Val Precision: {val_metrics['precision']:.4f} | "
        f"Val Recall: {val_metrics['recall']:.4f} | "
        f"Val F1: {val_metrics['f1']:.4f}"
    )

    if val_metrics["f1"] > best_val_f1:
        best_val_f1 = val_metrics["f1"]

        torch.save(
            model.state_dict(),
            MODEL_FILE,
        )


# ==================================================
# 8. LOAD THE BEST MODEL
# ==================================================

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=True,
    )
)

print("\nBest validation F1:", round(best_val_f1, 4))


# ==================================================
# 9. FINAL TEST EVALUATION
# ==================================================

test_metrics, y_true, y_pred = evaluate(
    X_test,
    y_test,
    test_edges,
)

print("\n" + "=" * 55)
print("FINAL TEST RESULTS")
print("=" * 55)

for name, value in test_metrics.items():
    print(f"{name.capitalize():12s}: {value:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        zero_division=0,
    )
)

print("Confusion matrix:")
print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )
)

print("\nSaved model:")
print(MODEL_FILE)

print("\nBaseline training completed.")
