# Courtesy of Long Range Graph Benchmarks (Dwivedi et. al, 2022) and https://github.com/vijaydwivedi75/lrgb

import torch
import torch.nn as nn
from torch_geometric.nn import MLP

from src.encoder.linEnc import LinearNodeEncoder, LinearEdgeEncoder
from src.layer.san2_layer import SAN2Layer
from src.layer.san_layer import SANLayer


class SAN(torch.nn.Module):
    """Spectral Attention Network (SAN) Graph Transformer.
    https://arxiv.org/abs/2106.03893
    """
    '''
    def __init__(self, input_channels, hidden_channels, num_layers, output_channels, heads, layerpm = 1,
                 layer_type='SANLayer', dropout=True, layer_norm = False, batch_norm = True, residual=True, head_type='default'):
    '''

    def __init__(self, node_in, edge_in, hidden_channels, num_layers, out_channels, heads = 4, gamma = 0, num_lin=2, pool=False, san2 = True, full_graph=False):
        super().__init__()
        self.enc = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.linlayers = nn.ModuleList()

        fake_edge_emb = torch.nn.Embedding(1, hidden_channels)

        node_enc = LinearNodeEncoder(node_in, hidden_channels)
        self.enc.append(node_enc)
        edge_enc = LinearEdgeEncoder(edge_in, hidden_channels)
        self.enc.append(edge_enc)
        for i in range(num_layers):
            start = hidden_channels
            end = hidden_channels
            if san2:
                self.layers.append(SAN2Layer(gamma, start, end, heads, full_graph, fake_edge_emb))
            else:
                self.layers.append(SANLayer(gamma, start, end, heads, full_graph, fake_edge_emb))
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

        self.pool = pool

    def forward(self, batch):
        # Graph Level Network
        if self.pool:
            h = []
            for i in range(len(self.layers)):
                if i == 0:
                    h.append(self.layers[i](batch))
                else:
                    h.append(self.layers[i](h[-1]))
                if i != len(self.layers) - 1:
                    h[i] = nn.functional.relu(h[i])
            h = [global_add_pool(i, batch) for i in h]
            h = torch.cat(h, dim=1)
            
            '''
            # Classifier
            for i in range(len(self.linlayers)):
                h = self.linlayers[i](h)
                if i != len(self.linlayers) - 1:
                    h = nn.functional.relu(h)
            '''
            # Dropout (uncomment if needed)
            # h = F.dropout(h, p=0.5, training=self.training)
            return h

        # Node level network
        for i in range(len(self.enc)):
            batch = self.enc[i](batch)
        for i in range(len(self.layers)):
            if i == 0:
                x = self.layers[i](batch)
            else:
                x = self.layers[i](x)
            if i != len(self.layers) - 1:
                batch.x = nn.functional.relu(batch.x)
                
        return batch

