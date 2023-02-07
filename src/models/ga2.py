import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool
from torch_geometric.utils import normalized_laplacian, add_self_loops, degree

class GraphAttention2(nn.Module):
    #prioritized by dist
    def __init__(self, in_channels, out_channels, heads=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.weight = nn.Parameter(torch.Tensor(heads, in_channels, out_channels))
        self.attention = nn.Parameter(torch.Tensor(heads, 2 * out_channels, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attention)

    def forward(self, x, edge_index, edge_dist):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        edge_index = edge_index.t().contiguous()
        edge_dist = edge_dist.view(-1, 1)

        out = x @ self.weight

        h = out.size(2)
        edge_index, edge_dist = gnn.sort_edge_index(edge_index, edge_dist)
        edge_index, _ = gnn.remove_self_loops(edge_index)
        edge_index, edge_dist = gnn.add_self_loops(edge_index, edge_dist, num_nodes=out.size(1))

        row, col = edge_index

        out = out[:, row]
        out = torch.cat([out[:, :, :h], out[:, :, :h]], dim=-1)
        out = out.view(-1, 2 * h)

        alpha = (out @ self.attention).view(-1)
        alpha = gnn.softmax(alpha, row, out.size(0))

        out = out * alpha.view(-1, 1)
        out = out.view(-1, self.heads, 2 * h).sum(dim=-1) / edge_dist
        out = out[row] - out[col]

        return out.view(-1, h)