import numpy as np
import torch
from torch.utils.data import Dataset

from project.utils.utils import rand_rot, get_fully_connected_graph


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


class TetrisDGLDataset(Dataset):
    node_feature_size = 1
    edge_feature_size = 1

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
