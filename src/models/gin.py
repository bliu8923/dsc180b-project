import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import MLP, GINConv, global_add_pool

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GINConv(MLP([in_channels, hidden_channels, hidden_channels], num_layers = 3))
        self.conv2 = GINConv(MLP([hidden_channels, hidden_channels, hidden_channels], num_layers = 3), train_eps=False)
        self.conv3 = GINConv(MLP([hidden_channels, hidden_channels, hidden_channels], num_layers = 3), train_eps=False)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, batch, pool):
        x = batch.x
        edge_index = batch.edge_index
        batch = batch.batch
        # Node embeddings
        if pool:
            h1 = self.conv1(x, edge_index).relu()
            h2 = self.conv2(h1, edge_index).relu()
            h3 = self.conv3(h2, edge_index)

            h1 = global_add_pool(h1, batch)
            h2 = global_add_pool(h2, batch)
            h3 = global_add_pool(h3, batch)

            h = torch.cat((h1, h2, h3), dim=1)

            # Classifier
            h = self.lin1(h)
            h = h.relu()
            h = F.dropout(h, p=0.5, training=self.training)
            h = self.lin2(h)

            return h
        else:
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()
            x = self.conv3(x, edge_index)

            h = self.lin1(x)
            h = h.relu()
            h = F.dropout(h, p=0.5, training=self.training)
            h = F.log_softmax(self.lin2(h))
            return h