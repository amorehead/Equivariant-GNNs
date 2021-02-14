import dgl
import numpy as np
import torch
from torch.utils.data import Dataset


class RGDGLDataset(Dataset):
    def __init__(self, mode='train', num_nodes=20, num_edges=10, node_feature_size=6, edge_feature_size=4,
                 size=300, dtype=np.float32):
        assert mode in ['train', 'val', 'test']
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.size = size
        self.dtype = dtype
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        y = np.array([np.random.random(1).astype(self.dtype)])
        # Create graph
        # G = dgl.DGLGraph((src,dst))
        G = dgl.rand_graph(self.num_nodes, self.num_edges)
        # Add node features to graph
        G.ndata['x'] = torch.tensor(np.random.random((self.num_nodes, 3)).astype(self.dtype))  # [num_atoms,3]
        G.ndata['f'] = torch.tensor(
            np.random.random((self.num_nodes, self.node_feature_size, 1)).astype(self.dtype))  # [num_atoms,6,1]
        # Add edge features to graph
        G.edata['d'] = torch.tensor(np.random.random((self.num_edges, 3)).astype(self.dtype))  # [num_atoms,3]
        G.edata['w'] = torch.tensor(
            np.random.random((self.num_edges, self.edge_feature_size)).astype(self.dtype))  # [num_atoms,4]
        return G, y
