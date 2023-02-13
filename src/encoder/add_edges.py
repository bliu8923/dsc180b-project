import torch_geometric.utils as tg

def add_edges(dataset, p = 0.5):
    try:
        dataset.data.edge_index, added_edges = tg.add_random_edge(dataset.data.edge_index, p)
    except:
        dataset.edge_index, added_edges = tg.add_random_edge(dataset.edge_index, p)
    return dataset, added_edges