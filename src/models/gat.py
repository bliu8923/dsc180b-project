import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GravNetConv
from torch_geometric.nn import global_add_pool, MLP


class GAT(torch.nn.Module):
    '''
    '''

    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, num_lin=2, pool=False, heads=4, dropout=0.0, partial=0, space=2, k=20):
        super().__init__()
        self.layers = nn.ModuleList()
        self.partials = nn.ModuleList()
        self.linlayers = nn.ModuleList()
        for i in range(num_layers):
            start = in_channels * heads
            end = hidden_channels
            if i == 0:
                start = in_channels
            self.layers.append(GATConv(in_channels=start, out_channels=end, num_layers=4, heads=heads, cached=True, normalize=True, dropout=dropout))
        for i in range(partial):
            start = in_channels
            end = hidden_channels
            if i == 0:
                if pool:
                    start = (hidden_channels * (num_layers)) * heads
                else:
                    start = hidden_channels * heads
            self.partials.append(GravNetConv(in_channels=start, out_channels=end, space_dimensions=space, propagate_dimensions=start - space, k=k))
        for i in range(num_lin):
            start = hidden_channels
            end = hidden_channels
            if i == 0:
                start = hidden_channels
            if i == num_lin - 1:
                end = out_channels
            self.linlayers.append(MLP(in_channels=start, hidden_channels=start, out_channels=end, num_layers=8))

        self.pool = pool

    def forward(self, batch):
        x = batch.x
        x = x.to(torch.float32)
        edge_index = batch.edge_index
        batch = batch.batch

        # Graph Level Network
        if self.pool:
            h = []
            for i in range(len(self.layers)):
                if i == 0:
                    h.append(self.layers[i](x, edge_index))
                    if i != len(self.layers) - 1:
                        h[i] = nn.functional.elu(h[i])
                else:
                    h.append(self.layers[i](h[-1], edge_index))
                    if i != len(self.layers) - 1:
                        h[i] = nn.functional.relu(h[i])
            h = [global_add_pool(i, batch) for i in h]
            h = torch.cat(h, dim=1)
            # Partial
            for i in range(len(self.partials)):
                h = self.partials[i](h)
                if i != len(self.partials) - 1:
                    h = nn.functional.relu(h)
            # Classifier
            for i in range(len(self.linlayers)):
                h = self.linlayers[i](h)
                if i != len(self.linlayers) - 1:
                    h = nn.functional.relu(h)

            # Dropout (uncomment if needed)
            # h = F.dropout(h, p=0.5, training=self.training)
            return h

        # Node level network
        for i in range(len(self.layers)):
            x = self.layers[i](x, edge_index)
            if i != len(self.layers) - 1:
                x = nn.functional.elu(x)

        # Partial
        for i in range(len(self.partials)):
            x = self.partials[i](x)
            if i != len(self.partials) - 1:
                x = nn.functional.relu(x)

        for i in range(len(self.linlayers)):
            x = self.linlayers[i](x)
            if i != len(self.linlayers) - 1:
                x = nn.functional.relu(x)

        return x
