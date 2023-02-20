import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as gnn
from torch_geometric.data import Data, Dataset, DataLoader

def parse_off(file_path):
    with open(file_path, 'r') as f:
        # Read the header and get the number of vertices and faces
        f.readline()#skip 1st line
        header = f.readline().strip().split(' ')
        num_vertices = int(header[0])
        num_faces = int(header[1])
        
        # Read the vertices
        vertices = []
        for i in range(num_vertices):
            vertex = list(map(float, f.readline().strip().split(' ')))
            vertices.append(vertex)
        vertices = torch.tensor(vertices)
        
        # Read the faces and build the edges
        edges = []
        for i in range(num_faces):
            face = list(map(int, f.readline().strip().split(' ')[1:]))
            for j in range(len(face)):
                edge = (face[j], face[(j+1)%len(face)])
                edges.append(edge)
        edges = torch.tensor(edges, dtype=torch.long)
        
        #Pad and trim to match dimensionality
        #num_nodes = max(edges.max().item() + 1, vertices.size(0))
        #new_x = torch.zeros((num_nodes, vertices.size(1)))
        #new_x[:vertices.size(0), :] = vertices
        
        return Data(x=vertices, edge_index=edges.transpose(0,1))

# Set the root folder that contains all the subfolders of .off files
root_folder = 'ModelNet10'

# Get the subfolder names and corresponding categorization labels
subfolder_names = [d+'/train' for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
# Encode Labels
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(np.array(subfolder_names).reshape(-1, 1))
labels = enc.transform(np.array(subfolder_names).reshape(-1, 1)).toarray() #ohe
labels = torch.argmax(torch.tensor(labels),dim=1) #numeric
#labels = [int(d.split('_')[0]) for d in subfolder_names]

# IN PROGRESS: Parsing all files and build a list of graph data objects
graphs = []
for subfolder_name, label in zip(subfolder_names, labels):
    subfolder_path = os.path.join(root_folder, subfolder_name)
    for filename in os.listdir(subfolder_path):
        if filename.endswith('.off'):#Check file type
            file_path = os.path.join(subfolder_path, filename)
            graph = parse_off(file_path)
            graph.y = torch.tensor([label])
            graphs.append(graph)

def repartition(max_num_node,target_dataset):
    # Reset the max num of nodes to cap te dataset
    idx = [i for i in range(len(target_dataset)) if dataset.num_nodes <= max_num_node]
    # build the appropriate subset
    subset = torch.utils.data.dataset.Subset(dataset, idx)
    return subset

# Create Train and Test sets
train_size = int(0.7 * len(graphs))
test_size = len(graphs) - train_size
train_set, test_set = torch.utils.data.random_split(graphs, [train_size, test_size])

train_set_short = repartition(2500,train_set)
test_set_short = repartition(2500,train_set)
"""
# To save and graph data object, use
torch.save(...)
# To load, use
train_loader = torch.load(...)
"""

train_loader = DataLoader(train_set, batch_size= 12, drop_last=True)
test_loader = DataLoader(test_set, batch_size= 12, drop_last=True)