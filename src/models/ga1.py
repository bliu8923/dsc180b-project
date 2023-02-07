import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool
from torch_geometric.utils import normalized_laplacian, add_self_loops, degree
class GraphAttention1(MessagePassing):
    # random selection
    def __init__(self, in_channels, out_channels, p_keep=0.5):
        super(GraphAttention1, self).__init__(aggr='add')  # use Add aggregation.
        self.lin = nn.Linear(in_channels, out_channels)
        self.p_keep = p_keep

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        x = self.lin(x)

        # Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Normalize edge weights.
        edge_weight = degree(edge_index[0], x.size(0), dtype=x.dtype)
        edge_weight = 1 / torch.sqrt(edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight)

        # Randomly select a portion of node pairs for attention calculation.
        random_mask = torch.rand(edge_index.size(1)) < self.p_keep
        edge_index, edge_weight = edge_index[:, random_mask], edge_weight[random_mask]

        return self.propagate(edge_index, x=x, edge_weight=edge_weight, batch=batch)

    def message(self, x_j, edge_weight):
        # Calculate attention scores for the selected node pairs.
        scores = (x_j * edge_weight).sum(dim=-1)
        scores = F.leaky_relu(scores)

        return x_j * scores.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out
