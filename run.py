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
from src.train import train
from src.test import test
from src.test_lite import test_lite
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

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

if sys.argv[1] == 'test':
    args = {'test': True}
    args['hidden'] = 32
else:

    parser.add_argument('--device_id', default=0, type=int,
                        help='the id of the gpu to use')
    # Model Related
    parser.add_argument('--model', default='san', type=str,
                        help='Model being used')
    parser.add_argument('--test', default=False, type=bool,
                        help='Test on smaller dataset for performance')

    # Data Related
    parser.add_argument('--bz', default=32, type=int,
                        help='batch size')
    parser.add_argument('--shuffle_data', default=True, type=bool,
                        help='Shuffle the data')
    parser.add_argument('--dataset', default='PascalVOC-SP', type=str,
                        help='Dataset to use (from Long Range Graph Benchmarks)')
    # feel free to add more augmentation/regularization related arguments

    # Other Choices & hyperparameters
    parser.add_argument('--epoch', default=25, type=int,
                        help='number of epochs')
    # for loss
    parser.add_argument('--criterion', default='cross_entropy', type=str,
                        help='which loss function to use')
    # for optimizer
    parser.add_argument('--optimizer', default='adam', type=str,
                        help='which optimizer to use')
    parser.add_argument('--lr', default=0.05, type=float,
                        help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')

    parser.add_argument('--accuracy_metric', default='macro_f1', type=str,
                        help='which metric to perform for classification accuracy')

    parser.add_argument('--add_edges', default=0, type=float,
                        help='ratio of added edges')
    parser.add_argument('--encode', default='none', type=str,
                        help='positional encoding')
    parser.add_argument('--encode_k', default=3, type=int,
                        help='number of positional encoding values to add')

    parser.add_argument('--task', default='node', type=str,
                        help='(node) (graph) classification')

    parser.add_argument('--trainsplit', default=0.6, type=float,
                        help='ratio for train split')
    parser.add_argument('--valsplit', default=0.2, type=float,
                        help='ratio for val split')
    parser.add_argument('--testsplit', default=0.2, type=float,
                        help='ratio for test split')

    parser.add_argument('--metric', default='macrof1', type=str,
                        help='accuracy metric')

    args = vars(parser.parse_args())


def main(args):
    if args['test']:
        normalize_features = False
        dataset = Planetoid(root='test/testdata', name='Cora')
        in_channels = dataset.num_features
        out_channels = dataset.num_classes
        print(test_lite(dataset, in_channels, in_channels, out_channels, modeltype='gcn'))
        print(test_lite(dataset, in_channels, in_channels, out_channels, modeltype='gin'))
        print(test_lite(dataset, in_channels, in_channels, out_channels, modeltype='gan'))
        print(test_lite(dataset, in_channels, in_channels, out_channels, modeltype='san'))
        return
    else:
        gc.collect()
        torch.cuda.empty_cache()
        normalize_features = False

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device: ' + str(device))

        # Set split datasets for task (Graph Task Splits Courtesy of GraphGPS)
        if args['encode'] == 'lap':
            print("Encoding with LapPE")
            transformpe = AddLaplacianEigenvectorPE(args['encode_k'])
            train_dataset = LRGBDataset(root='data', name=args['dataset'], split='train', transform=transformpe)
            val_dataset = LRGBDataset(root='data', name=args['dataset'], split='val', transform=transformpe)
            test_dataset = LRGBDataset(root='data', name=args['dataset'], split='test', transform=transformpe)
        else:
            print("No encoding")
            train_dataset = LRGBDataset(root='data', name=args['dataset'], split='train')
            val_dataset = LRGBDataset(root='data', name=args['dataset'], split='val')
            test_dataset = LRGBDataset(root='data', name=args['dataset'], split='test')

        criterion = weighted_cross_entropy

        if normalize_features:
            train_dataset.transform = T.NormalizeFeatures()
            val_dataset.transform = T.NormalizeFeatures()
            test_dataset.transform = T.NormalizeFeatures()
        if args['encode'] != 'none':
            in_channels = train_dataset.num_features + args['encode_k']
        else:
            in_channels = train_dataset.num_features
        out_channels = train_dataset.num_classes

        # Add dummy edges
        print("Dummy edge ratio: " + str(args['add_edges']))
        traindata, train_edge = add_edges(train_dataset, args['add_edges'])
        valdata, val_edge = add_edges(val_dataset, args['add_edges'])
        testdata, test_edge = add_edges(test_dataset, args['add_edges'])

        # Dataloaders
        train_loader = DataLoader(traindata, args['bz'], True)
        val_loader = DataLoader(valdata, args['bz'])
        test_loader = DataLoader(testdata, args['bz'])
        print("Encoding finished")

        # Set pooling based on task
        if args['task'] == 'graph':
            pool = True
        else:
            pool = False

        # Set model type for testing
        modeltype = args['model']
        modeltypes = ['gcn', 'gin', 'gan', 'san']
        '''Trains and tests the model type given (defaults to all models)'''
        if modeltype == 'gcn':
            model = GCN(in_channels, in_channels, 8, out_channels, attention=False, pool=pool)
        elif modeltype == 'gin':
            model = GIN(in_channels, in_channels, 8, out_channels)
        elif modeltype == 'gan':
            model = GAT(in_channels, in_channels, 8, out_channels)
        elif modeltype == 'san':
            model = SAN(in_channels, in_channels, 4, out_channels, 4)
        elif modeltype == 'gcn+a':
            model = GraphAttention1(in_channels, out_channels)
        elif modeltype == 'gcn+a2':
            model = GraphAttention2(in_channels, out_channels, heads=4)
        elif modeltype == 'all':
            results = {}
            for i in range(len(modeltypes)):
                results[modeltypes[i]] = test(traindata, valdata, testdata, in_channels, hidden_channels, out_channels,
                                              epochs, modeltypes[i])
            return results
        else:
            raise NameError("No model of " + modeltype + " found.")

        model = model.to(device)

        # Set weights
        try:
            weights = np.flip(
                compute_class_weight('balanced', classes=range(0, traindata.num_classes), y=traindata.data.y.tolist()),
                axis=0).copy()
        except:
            weights = None

        '''
        # Encode data (LapPE)
        if args['encode'] == 'lap':
            print("Encoding data using LapPE")
            traindata = lap_pe(traindata, device)
            valdata = lap_pe(valdata, device)
            testdata = lap_pe(testdata, device)
        else:
            print("No encoding")
        '''

        # Set optimizer
        optimizer = torch.optim.Adam([
            dict(params=model.parameters(), weight_decay=args['weight_decay'], momentum=args['momentum'])
        ], lr=args['lr'])

        # Set accuracy metric
        if args['metric'] == 'macrof1':
            metric = f1_score
        elif args['metric'] == 'ap':
            metric = average_precision_score

        for i in range(args['epoch']):
            loss, trainacc, acc = train(train_loader, val_loader, model, optimizer, criterion, device, metric)
            print("Epoch " + str(i) + ': ' + modeltype + ' loss: ' + str(loss) + ', train acc: ' + str(
                trainacc) + ', val acc: ' + str(acc))

        print("Test Acc: " + str(test(test_loader, metric)))


if __name__ == '__main__':
    main(args)

# feel free to add more arguments if necessary


# %%
