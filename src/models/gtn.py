import torch
from torch_geometric.nn import TransformerConv, LayerNorm, Sequential
from torch.nn.functional import elu
class GTN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=10):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads)
        self.ln1 = LayerNorm(hidden_channels*heads)
        self.conv2 = TransformerConv(hidden_channels * heads, out_channels, 1)
        self.ln2 = LayerNorm(out_channels)

    def forward(self, batch):
        x = batch.x
        x = batch.edge_data
        x = self.conv1(x, edge_index)
        x = self.ln1(x).relu()
        x = self.conv2(x, edge_index)
        x = self.ln2(x).relu()
        return x
