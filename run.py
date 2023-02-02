#!/usr/bin/env python
import sys
import json
import torch
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
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges
from sklearn.model_selection import train_test_split
import argparse
import gc

from torch_geometric.graphgym.register import register_config


from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.models.gnn import GCN
from torch_geometric.nn import GIN, GAT
#from src.models.gin import GIN
#from src.models.gat import GAT
from src.models.gtn import GTN
from src.models.san import SAN
from src.get_data import get_data
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
    parser.add_argument('--hidden', default=32, type=int,
                        help='Number of hidden channels')
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

    args = vars(parser.parse_args())

def train(loader, val, model, optimizer, device, weights=None):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = []
    trainscores = []
    scores = []
    for data in loader:
        data.to(device)
        optimizer.zero_grad()
        try:
            out = model(data.x, data.edge_index)
            print(out.shape)
            loss = criterion(out, data.y)
            _,pred = torch.max(out, 1)
        except:
            out = model(data)
            loss = criterion(out, data.y)
            _,pred = torch.max(F.softmax(out), 1)
        loss.backward()
        optimizer.step()
        total_loss.append(float(loss))
        trainscores.append(f1_score(data.y.cpu().tolist(), pred.cpu().tolist(), average='macro'))
    for data in val:
        data.to(device)
        model.eval()
        try:
            pred = model(data.x, data.edge_index)
            _,pred = torch.max(pred,1)
        except:
            pred = model(data)
            _,pred = torch.max(F.softmax(pred), 1)
        scores.append(f1_score(data.y.cpu().tolist(), pred.cpu().tolist(), average='macro'))
    return np.mean(total_loss), np.mean(trainscores), np.mean(scores)

def test(traindata, valdata, testdata, in_channels, hidden_channels, out_channels, args, epochs = 200, modeltype='all'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ' + str(device))
    modeltypes = ['gcn', 'gin', 'gan', 'san']
    '''Trains and tests the model type given (defaults to all models)'''
    if modeltype == 'gcn':
        #model = GCN(in_channels, hidden_channels, 8, out_channels)
        model = GCN(in_channels, hidden_channels, 8, out_channels, pos_enc=True, attention=False)
    elif modeltype == 'gin':
        model = GIN(in_channels, hidden_channels, 8, out_channels)
    elif modeltype == 'gan':
        model = GAT(in_channels, hidden_channels, 8, out_channels)
    elif modeltype == 'san':
        model = SAN(in_channels, hidden_channels, 4, out_channels, 4)
    elif modeltype == 'all':
        results = {}
        for i in range(len(modeltypes)):
            results[modeltypes[i]] = test(traindata, valdata, testdata, in_channels, hidden_channels, out_channels, epochs, modeltypes[i])
        return results
    else:
        raise NameError("No model of " + modeltype + " found.")


    try:
        weights = np.flip(compute_class_weight('balanced', classes=range(0,traindata.num_classes), y = traindata.data.y.tolist()), axis=0).copy()
    except:
        weights=None

    train_loader = DataLoader(traindata, 32, True)
    val_loader = DataLoader(valdata, 32)
    test_loader = DataLoader(testdata, 32)

    model = model.to(device)
    '''
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), weight_decay=5e-4, momentum=0.7)
    ], lr=0.001)
    '''
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), weight_decay = args['weight_decay'], momentum=args['momentum'])
    ], lr=args['lr'])

    for i in range(epochs):
        loss, trainacc, acc = train(train_loader, val_loader, model, optimizer, device, weights)
        print(modeltype + ' loss: ' + str(loss) + ', train acc: ' + str(trainacc) + ', val acc: ' + str(acc))

    model.eval()
    pred = []
    y = []
    for data in test_loader:
        data = data.to(device)
        if modeltype != 'san':
            pred = model(data.x, data.edge_index).argmax(dim=-1)
        else:
            pred = model(data)
        _,pred = torch.max(F.softmax(pred), 1)
        pred += pred.tolist()
        y += data.y.tolist()
    return f1_score(testdata.y, pred, average='macro')

def test_test(data, in_channels, hidden_channels, out_channels, epochs = 20, modeltype='all'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data[0].to(device)
    print('Device: ' + str(device))
    modeltypes = ['gcn', 'gin', 'gan', 'san']
    '''Trains and tests the model type given (defaults to all models)'''
    if modeltype == 'gcn':
        model = GCN(in_channels, hidden_channels, 8, out_channels)
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
        if modeltype != 'san':
            out = model(data.x, data.edge_index)
        else:
            out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()


    model.eval()
    if modeltype != 'san':
        pred = model(data.x, data.edge_index).argmax(dim=-1)
    else:
        pred = model(data)
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
        train_dataset = LRGBDataset(root='data', name=args['dataset'], split='train')
        val_dataset = LRGBDataset(root='data', name=args['dataset'], split='val')
        test_dataset = LRGBDataset(root='data', name=args['dataset'], split='test')
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
