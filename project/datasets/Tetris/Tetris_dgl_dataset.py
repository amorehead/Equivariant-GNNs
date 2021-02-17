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


def get_fully_connected_graph(pos, fill=0, dtype=np.float32):
    # pos :n by 3 np.array for xyz
    x = np.array(range(pos.shape[0]))
    src = np.repeat(x, x.shape[0])
    dst = np.tile(x, x.shape[0])
    flag = src != dst
    G = dgl.graph((src[flag], dst[flag]))
    G.ndata['x'] = pos
    G.ndata['f'] = torch.tensor(np.full((G.num_nodes(), 1, 1), fill).astype(dtype))
    G.edata['w'] = torch.tensor(np.full((G.num_edges(), 1), fill).astype(dtype))
    return G


def tetris(rand=False, fill=0, dtype=np.float32):
    pos = [
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # L
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # T
        [(0, 0, 0), (1, 1, 0), (2, 1, 0), (2, 1, 0)],  # zigzag
    ]

    label = np.eye(len(pos))
    dataset = [torch.tensor(np.array(x).astype(dtype)) for x in pos]
    if rand:
        dataset = [rand_rot(x) for x in dataset]
    g_list = [get_fully_connected_graph(x, fill, dtype) for x in dataset]

    return g_list, label


class TDataset(Dataset):
    def __init__(self, transform=None, fill=0, dtype=np.float32):
        g_list, label = tetris(fill=fill, dtype=dtype)
        self.g_list = g_list
        self.y = label.astype(dtype)
        self.transform = transform
        self.dtype = dtype

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        g_idx = self.g_list[idx]
        if self.transform:
            new_pos = self.transform(g_idx.ndata['x'])
            g_idx.ndata['x'] = torch.tensor(new_pos.astype(self.dtype))
        g_idx.edata['d'] = g_idx.ndata['x'][g_idx.edges()[1], :] - g_idx.ndata['x'][g_idx.edges()[0], :]
        return g_idx, self.y[[idx], :]
