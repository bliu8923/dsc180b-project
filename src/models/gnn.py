import torch
import numpy as np
from torch_geometric.nn import GCNConv, GatedGraphConv, global_mean_pool, global_add_pool
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import get_laplacian, to_torch_coo_tensor, dense_to_sparse
import torch_geometric.utils as tg
from tqdm import tqdm

class GCN(torch.nn.Module):
    '''
    2 layer convolutional graph neural network
    '''
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, pos_enc=False, attention=False, add_edge=0, num_lin=2, pool=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.linlayers = nn.ModuleList()
        for i in range(num_layers):
            start = hidden_channels
            end = hidden_channels
            if i == 0:
                start = in_channels
            if i == num_layers - 1:
                end = out_channels
            #self.layers.append(GCNConv(start, end, cached=False, normalize=True))
            self.layers.append(GatedGraphConv(end, 4, cached=True, normalize=True))
        for i in range(num_lin):
            start = hidden_channels
            end = hidden_channels
            if i == 0:
                start = (hidden_channels * (num_layers-1)) + out_channels
            if i == num_lin - 1:
                end = out_channels
            self.linlayers.append(nn.Linear(start, end))
        '''
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True,
                             normalize=False)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=True,
                             normalize=False)
        self.bn1 = gnn.BatchNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels, cached=True,
                             normalize=False)
        self.bn2 = gnn.BatchNorm(hidden_channels)
        '''

        self.pos_enc = pos_enc
        self.attention = attention
        self.attention_layer = nn.Linear(out_channels, 1)
        self.add_edge = add_edge
        self.pool = pool


    def forward(self, batch):
        pool = True
        x = batch.x
        x = x.to(torch.float32)
        edge_index = batch.edge_index
        edge_index, added_edges = tg.add_random_edge(edge_index, self.add_edge)
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
                if i != len(self.layers) - 1:
                    h = nn.functional.relu(h)
            
            #h = F.dropout(h, p=0.5, training=self.training)
            return h
        
        #Node level network
        if self.pos_enc:
            x = x + self.positional_encoding(x, batch.edge_index)
        for i in range(len(self.layers)):
            x = self.layers[i](x, edge_index)
            if i != len(self.layers) - 1:
                x = nn.functional.relu(x)

        if self.attention:
            attention_weights = F.softmax(self.attention_layer(x), dim=1)
            x = (x * attention_weights)
            #print(x.shape)
            #x = x.sum(dim=-2)
            return x
        else:
            return x

    def positional_encoding(self, x, edge_index):
        """
        Computes positional encoding for each node by the eigenvalues of the graph Laplacian matrix.
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lap = get_laplacian(edge_index, normalization="sym", num_nodes=x.size(0))[0]
        lap = tg.to_dense_adj(lap)#, size=x.size(0))
        eigvals = torch.linalg.eigvalsh(lap, UPLO='U')
        eigvals = eigvals.to(device)
        return x + eigvals.T

