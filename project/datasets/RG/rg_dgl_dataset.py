import dgl
import numpy as np
from torch.utils.data import Dataset

from project.utils.utils import get_graph


def get_rgraph(num_nodes, num_edges, node_feature_size, edge_feature_size, dtype):
    G = dgl.rand_graph(num_nodes, num_edges)
    src = G.edges()[0].numpy()
    dst = G.edges()[1].numpy()
    # Add node features to graph
    pos = np.random.random((num_nodes, 3))  # [num_atoms,3]
    node_features = np.random.random((num_nodes, node_feature_size, 1))  # [num_atoms,node_feature_size,1]
    # Add edge features to graph
    edge_features = np.random.random((num_edges, edge_feature_size))  # [num_atoms,edge_feature_size]
    return get_graph(src, dst, pos, node_features, edge_features, dtype, False, num_nodes=num_nodes)


class RGDGLDataset(Dataset):
    def __init__(self, n_lb=10, n_hb=20, e_lb=10, e_hb=15, node_feature_size=6, edge_feature_size=4,
                 size=300, out_dim=1, transform=None, dtype=np.float32):
        # Provided dataset parameters
        self.n_lb = n_lb
        self.n_hb = n_hb
        self.e_lb = e_lb
        self.e_hb = e_hb
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.size = size
        self.out_dim = out_dim
        self.transform = transform
        self.dtype = dtype

        # Generated dataset properties
        self.num_nodes = np.random.randint(self.n_lb, self.n_hb, self.size)
        self.num_edges = np.random.randint(self.e_lb, self.e_hb, self.size)
        self.g_list = [get_rgraph(self.num_nodes[i], self.num_edges[i],
                                  self.node_feature_size, self.edge_feature_size, self.dtype) for i in range(size)]
        self.y = np.random.random((self.size, self.out_dim)).astype(self.dtype)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        g_idx = self.g_list[idx]
        if self.transform:
            new_pos = self.transform(g_idx.ndata['x'], dtype=self.dtype)
            g_idx.ndata['x'] = new_pos
        g_idx.edata['d'] = g_idx.ndata['x'][g_idx.edges()[1], :] - g_idx.ndata['x'][g_idx.edges()[0], :]
        return g_idx, self.y[[idx], :]
