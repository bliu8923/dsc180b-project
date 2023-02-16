import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as gnn
from torch_geometric.data import Data, Dataset, DataLoader


def parse_off_ex(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        vertex_count, face_count = map(int, lines[1].split())
        vertex_lines = lines[2:vertex_count + 2]
        face_lines = lines[vertex_count + 2:]
        vertices = [list(map(float, line.split())) for line in vertex_lines]
        faces = [list(map(int, line.split()[1:])) for line in face_lines]
        return torch.tensor(vertices), torch.tensor(faces)


def off_to_data_ex(file_path):
    vertices, faces = parse_off(file_path)
    edge_index = torch.zeros((2, 3 * faces.shape[0]), dtype=torch.long)
    for i, face in enumerate(faces):
        edge_index[:, 3 * i:3 * (i + 1)] = torch.tensor([face, face]).T
    data = Data(x=vertices, edge_index=edge_index)
    return data


# ---------------------------------------------------------------

def parse_off(file_path):
    with open(file_path, 'r') as f:
        # Read the header and get the number of vertices and faces
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
                edge = (face[j], face[(j + 1) % len(face)])
                edges.append(edge)
        edges = torch.tensor(edges, dtype=torch.long)

        return Data(x=vertices, edge_index=edges)


"""
# To save and graph data object, use
torch.save(...)
# To load, use
train_loader = torch.load(...)
"""