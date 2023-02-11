import torch_geometric.utils as tg

def add_edges(dataset, p = 0.5):
    dataset.data.edge_index, added_edges = tg.add_random_edge(dataset.data.edge_index, p)
    return dataset, added_edges