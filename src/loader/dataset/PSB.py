
from torch_geometric.data import InMemoryDataset, Data, Dataset, DataLoader, download_url, extract_zip
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil

from torch_geometric.loader import DataLoader
from tqdm import tqdm

import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch_geometric.transforms as T

class PSB(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, split='train', edge_add=1):
        self.edgeadd = edge_add
        super().__init__(root, transform=None, pre_transform=None, pre_filter=None)
        self.data, self.slices = torch.load(root + '/psb/processed' + str(self.edgeadd) + '/' + split + '.pt')

    @property
    def raw_file_names(self):
        return ['./data/psb/psb_raw.zip']

    @property
    def processed_file_names(self):
        return ['./data/psb/processed' + str(self.edgeadd) + '/train.pt', './data/psb/processed' + str(self.edgeadd) + '/val.pt', './data/psb/processed' + str(self.edgeadd) + '/test.pt']

    @property
    def processed_paths(self):
        return ['./data/psb/processed' + str(self.edgeadd) + '/']

    '''
    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url('https://www.dropbox.com/s/nu6rq4guv1uysep/psb_raw.zip?dl=0',\
                            self.root + '/psb')
        extract_zip(path, self.root+'/psb/raw')
        os.rename(osp.join(self.root, 'psb/raw'))
        os.unlink(path)
    '''
    def make_labels(self, file_path, skip=0):
        fp = file_path
        skip_lines = skip  # default=0

        with open(fp, "r") as f:
            lines = f.readlines()

        # Organize the document by label
        # labels = re.findall(r'\b\w+\s\w+\b', text)
        current_label = None
        labels = {}

        for i, line in enumerate(lines):
            if i < skip_lines:
                continue

            # match labels
            match = re.search(r'^([^\d\s]+(?:\s+[^\d\s]+)*)', line)
            tokens = line.strip().split()

            if match:
                current_label = match.group(1).strip()
                labels[current_label] = []

            elif len(tokens) == 1:  # Instance index
                labels[current_label].append(int(tokens[0]))

        # Build a mapping from instance index to label
        mapping = {}
        for label, indices in labels.items():
            for index in indices:
                mapping[index] = label

        return mapping

    def parse_off(self, file_path):
        """
        input: the filepath of the .off file
        output: the Data object after parsing that file
        """
        with open(file_path, 'r') as f:
            # Read the header and get the number of vertices and faces
            f.readline()  # skip 1st line
            header = f.readline().strip().split(' ')
            num_vertices = int(header[0])
            num_faces = int(header[1])

            # Read the vertices
            vertices = []
            for i in range(num_vertices):
                vertex = list(map(float, f.readline().strip().split(' ')))
                vertices.append(vertex)
            vertices = torch.tensor(vertices)

            pos = torch.tensor(vertices, dtype=torch.float)
            graph = Data(x=vertices, pos=pos)
            knn_edge = T.KNNGraph(k=1 + self.edgeadd, cosine=False, force_undirected=True)
            graph = knn_edge(graph)
            '''
            # Read the faces and build the edges
            edges = []
            for i in range(num_faces):
                face = list(map(int, f.readline().strip().split(' ')[1:]))
                for j in range(len(face)):
                    edge = (face[j], face[(j + 1) % len(face)])
                    edges.append(edge)
            edges = torch.tensor(edges, dtype=torch.long)
            graph = Data(x=vertices, edge_index=edges.transpose(0,1), pos=pos)
            '''
            # Pad and trim to match dimensionality
            # num_nodes = max(edges.max().item() + 1, vertices.size(0))
            # new_x = torch.zeros((num_nodes, vertices.size(1)))
            # new_x[:vertices.size(0), :] = vertices

            # creating positional matrix
            # pos = []
            # for i in range(num_vertices):
            #    pos.append([float(x) for x in f.readline().split()])
            return graph

    def process(self):
        os.mkdir(self.processed_paths[0])
        for i in os.listdir(self.root + '/psb/raw'):
            if i.endswith('.cla'):
                print(i)
                fp = self.root + '/psb/raw/' + i
                # Set the root folder that contains all the subfolders of .off files
                root_folder = './data/psb/raw/raw_off/'

                out_dict = self.make_labels(fp)
                ulab = np.unique(list(out_dict.values()))

                #  Label Encoder
                le = LabelEncoder()
                le.fit(ulab.reshape(-1, 1))

                # Get a mapping of idx:lab and generate label class
                out_dict = self.make_labels(fp)
                ulab = np.unique(list(out_dict.values()))

                # Parsing all files and build a list of graph data objects
                graphs = []
                for filename in tqdm(os.listdir(root_folder)):
                    if filename.endswith('.off'):  # Check file type
                        file_index = int(filename.split(".")[0][1:])
                        if out_dict.get(file_index):  # check index included in classification
                            filepath = os.path.join(root_folder, filename)
                            graph = self.parse_off(filepath)
                            file_label = le.transform([out_dict[file_index]])
                            #file_label_enc = np.zeros(len(ulab))
                            #file_label_enc[file_label] = 1
                            #graph.y = torch.from_numpy(file_label_enc).long()
                            graph.y = torch.tensor(file_label[0])
                            # add edge_attr
                            dist_transform = T.Distance()
                            graph = dist_transform(graph)
                            if self.pre_transform is not None:
                                graph = self.pre_transform(graph)
                            # graph = knn_transform(graph)
                            graphs.append(graph)
                data, slices = self.collate(graphs)
                data.y = F.one_hot(data.y)
                torch.save((data, slices), self.processed_paths[0] + fp.split('/')[-1].split('.')[0] + '.pt')
                print("Saved to "+ self.processed_paths[0] + fp.split('/')[-1].split('.')[0] + '.pt')
