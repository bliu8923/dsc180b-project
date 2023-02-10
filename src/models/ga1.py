import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool
from torch_geometric.utils import add_self_loops, degree, coalesce
class GraphAttention1(MessagePassing):
    # random selection
    def __init__(self, in_channels, out_channels, p_keep=0.5):
        super(GraphAttention1, self).__init__(aggr='add')  # use Add aggregation.
        self.lin = nn.Linear(in_channels, out_channels)
        self.p_keep = p_keep

    def forward(self, batch):
        print(batch)
        x = batch.x
        edge_index = batch.edge_index
        try:
            edge_weight = batch.edge_attr
        except:
            edge_weight = degree(edge_index[0], x.size(0), dtype=x.dtype)
        x = self.lin(x)

        # Normalize edge weights.
        edge_weight = 1 / torch.sqrt(edge_weight)
        # Add self-loops to the adjacency matrix.
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight)
        # Randomly select a portion of node pairs for attention calculation.
        random_mask = torch.rand(edge_index.size(1)) < self.p_keep
        edge_index = edge_index[:, random_mask]
        edge_weight = edge_weight[random_mask]

        return self.propagate(edge_index, x=x, edge_weight=edge_weight, batch=batch)

    def message(self, x_j, edge_weight):
        # Calculate attention scores for the selected node pairs.
        print(x_j.shape)
        print(edge_weight.shape)
        scores = (x_j * edge_weight).sum(dim=-1)
        scores = F.leaky_relu(scores)

        return x_j * scores.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out