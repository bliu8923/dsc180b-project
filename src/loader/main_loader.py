import os
import torch
import torch_geometric.transforms as T
import numpy as np
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader

from src.encoder.add_edges import add_edges
from src.loader.parse_off import parse_off


def main_loader(args):
    if args['datatype'] == 'LRGB':
        train_dataset = LRGBDataset(root=args['path'], name=args['dataset'], split='train')
        train_dataset.data.edge_weight = torch.from_numpy(np.ones(train_dataset.data.edge_attr.shape[0]))
        val_dataset = LRGBDataset(root=args['path'], name=args['dataset'], split='val')
        test_dataset = LRGBDataset(root=args['path'], name=args['dataset'], split='test')
    else:
        # Set the root folder that contains all the subfolders of .off files
        root_folder = args['path'] + '/' + args['dataset']

        # Get the subfolder names and corresponding categorization labels
        subfolder_names = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
        labels = [int(d.split('_')[0]) for d in subfolder_names]

        # Parsing all files and build a list of graph data objects
        graphs = []
        for subfolder_name, label in zip(subfolder_names, labels):
            subfolder_path = os.path.join(root_folder, subfolder_name)
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.off'):
                    file_path = os.path.join(subfolder_path, filename)
                    graph = parse_off(file_path)
                    graph.y = torch.tensor([label])
                    graphs.append(graph)

        # Create Train and Test sets
        train_size = int(args['trainsplit'] * len(graphs))
        val_size = int(args['valsplit'] * len(graphs))
        test_size = len(graphs) - train_size - val_size
        train_dataset, valtest_set = torch.utils.data.random_split(graphs, [train_size, test_size+val_size])
        val_dataset, test_dataset = torch.utils.data.random_split(valtest_set, [val_size, test_size])
    if args['norm_feat']:
        train_dataset.transform = T.NormalizeFeatures()
        val_dataset.transform = T.NormalizeFeatures()
        test_dataset.transform = T.NormalizeFeatures()

    # Add dummy edges
    print("Dummy edge ratio: " + str(args['add_edges']))
    traindata, train_edge = add_edges(train_dataset, args['add_edges'])
    valdata, val_edge = add_edges(val_dataset, args['add_edges'])
    testdata, test_edge = add_edges(test_dataset, args['add_edges'])

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

    return traindata, valdata, testdata