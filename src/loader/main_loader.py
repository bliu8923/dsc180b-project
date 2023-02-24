import os
import torch
import torch_geometric.transforms as T
import numpy as np
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader

from src.encoder.add_edges import add_edges
from src.loader.parse_off import parse_off
from src.loader.dataset.PSB import PSB

def main_loader(args):
    if args['datatype'] == 'LRGB':
        train_dataset = LRGBDataset(root=args['path'], name=args['dataset'], split='train')
        train_dataset.data.edge_weight = torch.from_numpy(np.ones(train_dataset.data.edge_attr.shape[0]))
        val_dataset = LRGBDataset(root=args['path'], name=args['dataset'], split='val')
        val_dataset.data.edge_weight = torch.from_numpy(np.ones(val_dataset.data.edge_attr.shape[0]))
        test_dataset = LRGBDataset(root=args['path'], name=args['dataset'], split='test')
        test_dataset.data.edge_weight = torch.from_numpy(np.ones(test_dataset.data.edge_attr.shape[0]))

        # Add dummy edges
        print("Dummy edge ratio: " + str(args['add_edges']))
        traindata, train_edge = add_edges(train_dataset, args['add_edges'])
        valdata, val_edge = add_edges(val_dataset, args['add_edges'])
        testdata, test_edge = add_edges(test_dataset, args['add_edges'])

    else:
        knn = T.KNNGraph(k=args['add_edges'])
        #train_dataset = PSB(root=args['path'], split='train', edge_num=args['add_edges'])
        traindata = PSB(root=args['path'], split='train', edge_add=round(args['add_edges']))
        #train_dataset.data.edge_weight = torch.from_numpy(np.ones(train_dataset.edge_attr.shape[0]))
        #val_dataset = PSB(root=args['path'], split='val', edge_num=args['add_edges'])
        valdata = PSB(root=args['path'], split='val', edge_add=round(args['add_edges']))
        #val_dataset.data.edge_weight = torch.from_numpy(np.ones(val_dataset.edge_attr.shape[0]))
        #test_dataset = PSB(root=args['path'], split='test', edge_num=args['add_edges'])
        testdata = PSB(root=args['path'], split='test', edge_add=round(args['add_edges']))
        #test_dataset.data.edge_weight = torch.from_numpy(np.ones(test_dataset.edge_attr.shape[0]))


    if args['norm_feat']:
        train_dataset.transform = T.NormalizeFeatures()
        val_dataset.transform = T.NormalizeFeatures()
        test_dataset.transform = T.NormalizeFeatures()


    if args['encode'] == 'lap':
        transform = T.AddLaplacianEigenvectorPE(args['encode_k'], attr_name=None)
        traindata.data = transform(traindata.data)
        valdata.data = transform(valdata.data)
        testdata.data = transform(testdata.data)
    if args['encode'] == 'walk':
        transform = T.AddRandomWalkPE(args['encode_k'], attr_name=None)
        traindata.data = transform(traindata.data)
        valdata.data = transform(valdata.data)
        testdata.data = transform(testdata.data)

    print(traindata.data)
    print(len(traindata.data))

    return traindata, valdata, testdata