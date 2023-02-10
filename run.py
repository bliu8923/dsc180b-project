#!/usr/bin/env python
import sys
import json
import torch
from tqdm import tqdm
import numpy as np
from torch import Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.datasets import LRGBDataset
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.logging import init_wandb, log
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.graphgym.loader import set_dataset_attr
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges, add_random_edge
from sklearn.model_selection import ShuffleSplit
import argparse
import gc

from torch_geometric.graphgym.register import register_config


from sklearn.metrics import f1_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight

from src.models.gnn import GCN
from torch_geometric.nn import GIN, GAT
#from src.models.gin import GIN
#from src.models.gat import GAT
from src.models.gtn import GTN
from src.models.san import SAN
from src.get_data import get_data
from src.models.ga1 import GraphAttention1
from src.models.ga2 import GraphAttention2
from src.loss.weighted_ce import weighted_cross_entropy

#SUPPRESSING WARNINGS FOR AP
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()

if sys.argv[1] == 'test':
    args = {'test':True}
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

def train(loader, val, model, optimizer, device, metric, weights=None):
    model.to(device)
    model.train()
    criterion = weighted_cross_entropy
    total_loss = []
    trainscores = []
    scores = []
    
    for data in tqdm(loader):
        model.train()
        data.to(device)
        optimizer.zero_grad()
        
        try:
            out = model(data.x, data.edge_index)
        except:
            out = model(data)
            
        loss, pred_score = criterion(out, data.y)
        if metric == f1_score:
            _, pred = torch.max(F.softmax(out, dim=1), 1)
        elif metric == average_precision_score:
            pred = F.softmax(out, dim=1)
        loss.backward()
        optimizer.step()
        total_loss.append(float(loss))
        trainscores.append(metric(data.y.cpu().tolist(), pred.cpu().tolist(), average='macro'))

    for data in tqdm(val):
        model.eval()
        data.to(device)
        try:
            pred = model(data.x, data.edge_index)
            _,pred = torch.max(pred,1)
        except:
            pred = model(data)
            if metric == f1_score:
                _, pred = torch.max(F.softmax(pred, dim=1), 1)
            elif metric == average_precision_score:
                pred = F.softmax(pred, dim=1)
        scores.append(metric(data.y.cpu().tolist(), pred.cpu().tolist(), average='macro'))
    return np.mean(total_loss), np.mean(trainscores), np.mean(scores)

def test(traindata, valdata, testdata, in_channels, hidden_channels, out_channels, args, epochs = 200, modeltype='all'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ' + str(device))
    
    #Set pooling based on task
    if args['task'] == 'graph':
        pool = True
    else:
        pool = False

    #Set model type for testing
    modeltypes = ['gcn', 'gin', 'gan', 'san']
    '''Trains and tests the model type given (defaults to all models)'''
    if modeltype == 'gcn':
        model = GCN(in_channels, in_channels, 8, out_channels, pos_enc=False, attention=False, add_edge=args['add_edges'], pool=pool)
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
            results[modeltypes[i]] = test(traindata, valdata, testdata, in_channels, hidden_channels, out_channels, epochs, modeltypes[i])
        return results
    else:
        raise NameError("No model of " + modeltype + " found.")

    model = model.to(device)

    #Set weights
    try:
        weights = np.flip(compute_class_weight('balanced', classes=range(0,traindata.num_classes), y = traindata.data.y.tolist()), axis=0).copy()
    except:
        weights=None
    
    #Dataloaders
    train_loader = DataLoader(traindata, args['bz'], True)
    val_loader = DataLoader(valdata, args['bz'])
    test_loader = DataLoader(testdata, args['bz'])
    
    
    #Set optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), weight_decay = args['weight_decay'], momentum=args['momentum'])
    ], lr=args['lr'])
    
    #Set accuracy metric
    if args['metric'] == 'macrof1':
        metric = f1_score
    elif args['metric'] == 'ap':
        metric = average_precision_score
    
    for i in range(epochs):
        loss, trainacc, acc = train(train_loader, val_loader, model, optimizer, device, metric, weights)
        print("Epoch " + str(i) + ': ' + modeltype + ' loss: ' + str(loss) + ', train acc: ' + str(trainacc) + ', val acc: ' + str(acc))

    model.eval()
    scores = []
    for data in tqdm(test_loader):
        data = data.to(device)
        try:
            pred = model(data.x, data.edge_index).argmax(dim=-1)
        except:
            pred = model(data)
        if metric == f1_score:
            _, pred = torch.max(F.softmax(pred, dim=1), 1)
        elif metric == average_precision_score:
            pred = F.softmax(pred, dim=1)
        scores.append(metric(data.y.cpu().tolist(), pred.cpu().tolist(), average='macro'))
    return np.mean(scores)

def test_test(data, in_channels, hidden_channels, out_channels, epochs = 20, modeltype='all'):
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


def main(args):

    if args['test']:
        normalize_features = False
        dataset = Planetoid(root='test/testdata', name='Cora')
        in_channels = dataset.num_features
        out_channels = dataset.num_classes
        print(test_test(dataset, in_channels, args['hidden'], out_channels, modeltype='gcn'))
        print(test_test(dataset, in_channels, args['hidden'], out_channels, modeltype='gin'))
        print(test_test(dataset, in_channels, args['hidden'], out_channels, modeltype='gan'))
        print(test_test(dataset, in_channels, args['hidden'] , out_channels, modeltype='san'))
        return
    else:
        gc.collect()
        torch.cuda.empty_cache()
        normalize_features = False

        # Set split datasets for task (Graph Task Splits Courtesy of GraphGPS)
        task_level = args['task']
        train_dataset = LRGBDataset(root='data', name=args['dataset'], split='train')
        val_dataset = LRGBDataset(root='data', name=args['dataset'], split='val')
        test_dataset = LRGBDataset(root='data', name=args['dataset'], split='test')
        '''
        if task_level == 'node':
            train_dataset = LRGBDataset(root='data', name=args['dataset'], split='train')
            val_dataset = LRGBDataset(root='data', name=args['dataset'], split='val')
            test_dataset = LRGBDataset(root='data', name=args['dataset'], split='test')
        elif task_level == 'graph':
            dataset = LRGBDataset(root='data', name=args['dataset'], split='train')
            val_dataset = LRGBDataset(root='data', name=args['dataset'], split='val')
            test_dataset = LRGBDataset(root='data', name=args['dataset'], split='test')
            n1, n2, n3 = len(dataset), len(val_dataset), len(test_dataset)
            data_list = [dataset.get(i) for i in range(n1)] + \
                        [val_dataset.get(i) for i in range(n2)] + \
                        [test_dataset.get(i) for i in range(n3)]

            dataset._indices = None
            dataset._data_list = data_list
            dataset.data, dataset.slices = dataset.collate(data_list)

            split_names = [
                'train_graph_index', 'val_graph_index', 'test_graph_index'
            ]
            split_ratios = [args['trainsplit'], args['valsplit'], args['testsplit']]
            train_index, val_test_index = next(
                ShuffleSplit(
                    train_size=split_ratios[0],
                ).split(dataset.data.y, dataset.data.y)
            )
            val_index, test_index = next(
                ShuffleSplit(
                    train_size=split_ratios[1] / (1 - split_ratios[0]),
                ).split(dataset.data.y[val_test_index], dataset.data.y[val_test_index])
            )
            val_index = val_test_index[val_index]
            test_index = val_test_index[test_index]
            print(train_index, val_index, test_index)
            for split_name, split_index in zip(split_names, [train_index, val_index, test_index]):
                set_dataset_attr(dataset, split_name, split_index, 3)
            print(dataset.data)
            train_dataset = dataset[train_index]
            print(train_dataset)
            val_dataset = dataset[val_index]
            print(val_dataset)
            test_dataset = dataset[test_index]
            print(test_dataset)

        else:
            raise ValueError(f"Unsupported dataset task level: {task_level}")
        '''
        if normalize_features:
            train_dataset.transform = T.NormalizeFeatures()
            val_dataset.transform = T.NormalizeFeatures()
            test_dataset.transform = T.NormalizeFeatures()
        in_channels = train_dataset.num_features
        out_channels = train_dataset.num_classes
        print(test(train_dataset, val_dataset, test_dataset, in_channels, 8, out_channels, args, args['epoch'], args['model']))



if __name__ == '__main__':
    main(args)


# feel free to add more arguments if necessary


#%%
