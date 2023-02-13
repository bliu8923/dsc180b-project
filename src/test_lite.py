#!/usr/bin/env python
import argparse
import gc
import json
import sys
# SUPPRESSING WARNINGS FOR AP
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import f1_score, average_precision_score, accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from src.encoder.add_edges import add_edges
from src.encoder.lapPE import lap_pe
from src.get_data import get_data
from src.loss.weighted_ce import weighted_cross_entropy
from src.loss.cross_entropy import multilabel_cross_entropy
from src.models.ga1 import GraphAttention1
from src.models.ga2 import GraphAttention2
from src.models.gnn import GCN
from src.models.gin import GIN
from src.models.gat import GAT
from src.models.san import SAN
from torch import Tensor
from torch.nn import Linear, Parameter
from torch_geometric.datasets import LRGBDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.graphgym.loader import set_dataset_attr
from torch_geometric.graphgym.register import register_config
from torch_geometric.loader import DataLoader
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.utils import train_test_split_edges, add_random_edge
from tqdm import tqdm

def test_lite(dataset, in_channels, hidden_channels, out_channels, epochs = 20, modeltype='all', bz=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    
    train_loader = DataLoader(dataset, bz, True)
    test_loader = DataLoader(dataset, bz)
    
    print('Device: ' + str(device) + ', Model: ' + str(modeltype))
    modeltypes = ['gcn', 'gin', 'gan', 'san']
    '''Trains and tests the model type given (defaults to all models)'''
    if modeltype == 'gcn':
        model = GCN(in_channels, hidden_channels, 4, out_channels)
    elif modeltype == 'gin':
        model = GIN(in_channels, hidden_channels, 4, out_channels)
    elif modeltype == 'gan':
        model = GAT(in_channels, hidden_channels, 2, out_channels)
    elif modeltype == 'san':
        model = SAN(in_channels, data.edge_index.shape[0], 2000, 2, out_channels, 4)
    else:
        print("No model found")
        return None


    model = model.to(device)

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), momentum=0.9)
    ], lr=0.0005)

    criterion = weighted_cross_entropy
    
    losses = []
    for i in range(epochs):
        for data in train_loader:
            data.edge_attr = torch.from_numpy(np.zeros_like((data.edge_index.shape[-1], 1))).to(device)
            data.edge_attr = data.edge_attr.type(torch.float32)
            tdata = data.clone()
            tdata = tdata.to(device)
            model.train()
            optimizer.zero_grad()
            if type(model) == SAN:
                out = model(tdata).x
            else:
                out = model(tdata)
            if criterion == weighted_cross_entropy:
                loss, pred = criterion(out[tdata.train_mask], tdata.y[tdata.train_mask])
            else:
                loss = criterion(out[tdata.train_mask], tdata.y[tdata.train_mask])
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
    
    print("Loss values (Check for convergence): " + str(losses))

    model.eval()
    for data in test_loader:
        data.edge_attr = torch.from_numpy(np.zeros_like((data.edge_index.shape[-1], 1))).to(device)
        data.edge_attr = data.edge_attr.type(torch.float32)
        data = data.to(device)
        if type(model) == SAN:
            pred = model(data).x
        else:
            pred = model(data)
        _, pred = torch.max(F.log_softmax(pred), 1)
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            #Accuracy score
            accs.append(accuracy_score(data.y[mask].cpu().tolist(), pred[mask].cpu().tolist()))
    return accs