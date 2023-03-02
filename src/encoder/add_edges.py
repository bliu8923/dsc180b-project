import torch_geometric.utils as tg
import numpy as np
import torch
def add_edges(dataset, p = 0.5):
    try:
        dataset.data.edge_index, added_edges = tg.add_random_edge(dataset.data.edge_index, p)
        if len(dataset.data.edge_attr) > 0:
            val = torch.mean(dataset.data.edge_attr, dtype=torch.float32).long().item()
            dataset.data.edge_attr = torch.cat((dataset.data.edge_attr, torch.from_numpy(np.full((added_edges.shape[1],dataset.data.edge_attr.shape[1]), val))), dim=0)
        if len(dataset.data.edge_weight) > 0:
            val = torch.mean(dataset.data.edge_weight, dtype=torch.float64).long().item()
            dataset.data.edge_weight = torch.cat((dataset.data.edge_weight, torch.from_numpy(np.full(added_edges.shape[1], val))), dim=0)
            print(dataset.data.edge_index.shape)
            print(dataset.data.edge_weight.shape)
    except:
        dataset.edge_index, added_edges = tg.add_random_edge(dataset.edge_index, p)
        if len(dataset.edge_attr) > 0:
            val = torch.mean(dataset.edge_attr, dtype=torch.float32).long().item()
            dataset.edge_attr = torch.cat(dataset.edge_attr, torch.from_numpy(
                np.full((len(added_edges), dataset.edge_attr.shape[1]), val)))
        if len(dataset.edge_weight) > 0:
            val = torch.mean(dataset.edge_weight, dtype=torch.float64).long().item()
            dataset.edge_weight = torch.cat((dataset.edge_weight, torch.from_numpy(np.full(added_edges.shape[1], val))), dim=0)
            print(dataset.edge_index.shape)
            print(dataset.edge_weight.shape)

    return dataset, added_edges