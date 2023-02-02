from torch_geometric.datasets import LRGBDataset

def get_data(dataset):
    if dataset == 'PascalVOC':
        return LRGBDataset(root='data', name='PascalVOC-SP')
    elif dataset == 'COCO':
        return LRGBDataset(root='data', name='COCO-SP')
    elif dataset == 'PCQM':
        return LRGBDataset(root='data', name='PCQM-Contact')
    elif dataset == 'Peptides-func':
        return LRGBDataset(root='data', name='Peptides-func')
    elif dataset == 'Peptides-struct':
        return LRGBDataset(root='data', name='Peptides-struct')
    else:
        print("No data found")
        return 0
#%%
