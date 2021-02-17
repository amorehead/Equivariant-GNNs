import dgl
import numpy as np
import torch
from torch.utils.data import Dataset

def get_graph(src, dst, pos, node_feature, edge_feature, dtype, undirected=True, num_nodes=None):
    # src, dst : indices for vertices of source and destination, np.array
    # pos: x,y,z coordinates of all vertices with respect to the indices, np.array
    # node_feature: node feature of shape [num_atoms,node_feature_size,1], np.array
    # edge_feature: edge feature of shape [num_atoms,edge_feature_size], np.array
    if num_nodes:
        G = dgl.graph((src, dst), num_nodes=num_nodes)
    else:
        G = dgl.graph((src, dst))
    if undirected:
        G = dgl.to_bidirected(G)
    # Add node features to graph
    G.ndata['x'] = torch.tensor(pos.astype(dtype))  # [num_atoms,3]
    G.ndata['f'] = torch.tensor(node_feature.astype(dtype))
    # Add edge features to graph
    G.edata['w'] = torch.tensor(edge_feature.astype(dtype))  # [num_atoms,edge_feature_size]
    return G

def get_rgraph(num_node, num_edge, node_feature_size, edge_feature_size, dtype):
    G = dgl.rand_graph(num_node, num_edge)
    src = G.edges()[0].numpy()
    dst = G.edges()[1].numpy()
    # Add node features to graph
    pos = np.random.random((num_node, 3))  # [num_atoms,3]
    node_feature = np.random.random((num_node, node_feature_size, 1))  # [num_atoms,node_feature_size,1]
    # Add edge features to graph
    edge_feature = np.random.random((num_edge, edge_feature_size))  # [num_atoms,edge_feature_size]
    return get_graph(src, dst, pos, node_feature, edge_feature, dtype, False, num_nodes=num_node)

class RGDGLDataset(Dataset):
    def __init__(self, n_lb=10, n_hb=20, e_lb=10, e_hb=15, node_feature_size=6, edge_feature_size=4,
                 size=300, out_dim=1, transform=None, dtype=np.float32):
        num_nodes = np.random.randint(n_lb, n_hb, size)
        num_edges = np.random.randint(e_lb, e_hb, size)
        self.g_list = [get_rgraph(num_nodes[i], num_edges[i], node_feature_size, edge_feature_size, dtype) for i in
                       range(size)]
        self.y = np.random.random((size, out_dim)).astype(dtype)
        self.transform = transform
        self.dtype = dtype
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        g_idx = self.g_list[idx]
        if self.transform:
            new_pos = self.transform(g_idx.ndata['x'])
            g_idx.ndata['x'] = torch.tensor(new_pos.astype(self.dtype))
        g_idx.edata['d'] = g_idx.ndata['x'][g_idx.edges()[1], :] - g_idx.ndata['x'][g_idx.edges()[0], :]
        return g_idx, self.y[[idx], :]
