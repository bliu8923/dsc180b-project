import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GatedGraphConv, global_add_pool, MLP


class GCN(torch.nn.Module):
    '''
    2 layer convolutional graph neural network
    '''
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, attention=False, num_lin=2, pool=False, gated=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.linlayers = nn.ModuleList()
        for i in range(num_layers):
            start = hidden_channels
            end = hidden_channels
            if i == 0:
                start = in_channels
            if gated:
                self.layers.append(GatedGraphConv(start, end, cached=True, normalize=True))
            else:
                self.layers.append(GCNConv(start, end, cached=False, normalize=True))
        for i in range(num_lin):
            start = hidden_channels
            end = hidden_channels
            if i == 0:
                if pool:
                    start = (hidden_channels * (num_layers))
                else:
                    start = hidden_channels
            if i == num_lin - 1:
                end = out_channels
            self.linlayers.append(MLP(in_channels=start, hidden_channels=start, out_channels=end, num_layers=8))

        self.attention = attention
        self.attention_layer = nn.Linear(out_channels, 1)
        self.pool = pool


    def forward(self, batch):
        x = batch.x
        x = x.to(torch.float32)
        edge_index = batch.edge_index
        batch = batch.batch
        
        #Graph Level Network
        if self.pool:
            h = []
            for i in range(len(self.layers)):
                if i == 0:
                    h.append(self.layers[i](x, edge_index))
                    if i != len(self.layers) - 1:
                        h[i] = nn.functional.relu(h[i])
                else:
                    h.append(self.layers[i](h[-1], edge_index))
                    if i != len(self.layers) - 1:
                        h[i] = nn.functional.relu(h[i])
            h = [global_add_pool(i, batch) for i in h]
            h = torch.cat(h, dim=1)
            # Classifier
            for i in range(len(self.linlayers)):
                h = self.linlayers[i](h)
                if i != len(self.linlayers) - 1:
                    h = nn.functional.relu(h)

            #Dropout (uncomment if needed)
            #h = F.dropout(h, p=0.5, training=self.training)
            return h
        
        #Node level network
        for i in range(len(self.layers)):
            x = self.layers[i](x, edge_index)
            if i != len(self.layers) - 1:
                x = nn.functional.relu(x)

        for i in range(len(self.linlayers)):
            x = self.linlayers[i](x)
            if i != len(self.linlayers) - 1:
                x = nn.functional.relu(x)
                
        if self.attention:
            attention_weights = F.softmax(self.attention_layer(x), dim=1)
            x = (x * attention_weights)
            return x
        else:
            return x

