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
from sklearn.metrics import f1_score, average_precision_score
from sklearn.model_selection import ShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from src.encoder.add_edges import add_edges
from src.encoder.lapPE import lap_pe
from src.get_data import get_data
from src.loss.weighted_ce import weighted_cross_entropy
from src.models.ga1 import GraphAttention1
from src.models.ga2 import GraphAttention2
from src.models.gnn import GCN
# from src.models.gin import GIN
# from src.models.gat import GAT
from src.models.gtn import GTN
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
from torch_geometric.nn import GIN, GAT
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.utils import train_test_split_edges, add_random_edge
from tqdm import tqdm

def test_lite(data, in_channels, hidden_channels, out_channels, epochs = 20, modeltype='all'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data[0].to(device)
    print('Device: ' + str(device))
    modeltypes = ['gcn', 'gin', 'gan', 'san']
    '''Trains and tests the model type given (defaults to all models)'''
    if modeltype == 'gcn':
        model = GCN(in_channels, hidden_channels, 8, out_channels, True, True)
    elif modeltype == 'gin':
        model = GIN(in_channels, hidden_channels, 8, out_channels)
    elif modeltype == 'gan':
        model = GAT(in_channels, hidden_channels, 8, out_channels)
    elif modeltype == 'san':
        print("SAN can't run on CORA due to edge attr limitations. If you want to try running SAN architectures, use a " +\
              "LRGB dataset")
        return None
    else:
        print("No model found")
        return None


    model = model.to(device)

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), momentum=0.9)
    ], lr=0.01)

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0
    total_correct = 0
    for i in range(epochs):
        model.train()
        optimizer.zero_grad()
        try:
            out = model(data.x[data.train_mask], data.edge_index[data.train_mask])
        except:
            out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()


    model.eval()
    try:
        pred = model(data.x, data.edge_index).argmax(dim=-1)
    except:
        pred = model(data)
        _, pred = torch.max(F.softmax(pred), 1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(f1_score(data.y[mask].cpu().tolist(), pred[mask].cpu().tolist(), average='macro'))
    return accs