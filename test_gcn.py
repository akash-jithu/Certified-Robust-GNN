
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# A tiny graph with 4 nodes and 4 directed edges
edge_index = torch.tensor([
    [0, 1, 2, 3],
    [1, 0, 3, 2]
], dtype=torch.long)

# Each node has 3 input features
x = torch.tensor([
    [1.0, 0.0, 0.5],
    [0.0, 1.0, 0.5],
    [1.0, 1.0, 0.0],
    [0.5, 0.5, 1.0]
])

# Create a GCN layer: 3 input features -> 4 output features
gcn = GCNConv(in_channels=3, out_channels=4)

# Run the graph through the GCN
output = gcn(x, edge_index)
output = F.relu(output)

print("Input shape:", x.shape)
print("Edge index shape:", edge_index.shape)
print("GCN output shape:", output.shape)
print("\nGCN output:")
print(output)

assert output.shape == (4, 4)
print("\nSUCCESS: GCN layer is working!")
