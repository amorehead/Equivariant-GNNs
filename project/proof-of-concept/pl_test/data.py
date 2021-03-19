import torch
import dgl
import numpy as np

from torch.utils.data import Dataset
from dgl.data import DGLDataset


def generate_random_dict(data_types, min_len=30, max_len=50, order=5):
    x_len = np.random.randint(min_len, max_len)
    sample = {}
    if 'one_hot' in data_types:
        sample['one_hot'] = torch.tensor(np.random.randint(0, 2, (x_len, 20)).astype(np.float32))
    if 'position' in data_types:
        sample['position'] = torch.tensor(np.random.random((x_len, 3)).astype(np.float32))
    if 'features' in data_types:
        sample['features'] = torch.tensor(np.random.random((x_len, 3)).astype(np.float32))
    if 'sh' in data_types:
        sample['sh'] = [torch.tensor(np.random.random((x_len, x_len)).astype(np.float32)) for i in
                        range(np.square(order))]
    if 'cad' in data_types:
        sample['cad'] = torch.tensor(np.random.random(x_len).astype(np.float32))
    if 'lddt' in data_types:
        sample['lddt'] = torch.tensor(np.random.random(x_len).astype(np.float32))
        sample['lddt_mask'] = torch.tensor(np.random.randint(0, 2, x_len).astype(np.bool_))
    return sample


def dict2dgl(data_x, dist_cutoff=15, neighbor_val=1, max_edge=5000, step=0.1):
    dist_map = torch.cdist(data_x['position'], data_x['position']).squeeze()
    cmap = dist_map <= dist_cutoff
    while torch.sum(cmap) > max_edge and dist_cutoff > 0:
        dist_cutoff -= step
        cmap = dist_map <= dist_cutoff
    el = torch.where(cmap)
    ew = torch.abs(el[0] - el[1]) > neighbor_val
    u = el[0][ew]
    v = el[1][ew]
    g = dgl.graph((u, v), num_nodes=dist_map.shape[0])
    g.ndata['x'] = data_x['position'].squeeze()
    g.ndata['f'] = torch.cat((data_x['one_hot'].squeeze(),
                              data_x['features'].squeeze()), 1).type_as(data_x['position'])
    g.ndata['f'] = torch.reshape(g.ndata['f'], (g.ndata['f'].shape[0],
                                                g.ndata['f'].shape[1], 1))
    u, v = g.edges()
    ew = torch.abs(u - v)
    ed = dist_map[u, v]
    g.edata['w'] = torch.stack((ed, ew)).type_as(data_x['position']).T
    g.edata['d'] = g.ndata['x'][g.edges()[1], :] - g.ndata['x'][g.edges()[0], :]
    if 'cad' in data_x.keys():
        g.ndata['cad'] = data_x['cad'].squeeze()
    if 'lddt' in data_x.keys():
        g.ndata['lddt'] = data_x['lddt'].squeeze()
        g.ndata['lddt_mask'] = data_x['lddt_mask'].squeeze()
    return g


class Random_dgl_Dataset(DGLDataset):

    def __init__(self, n, data_types='one_hot,position,features,lddt,cad,sh',
                 dist_cutoff=0.1, neighbor_val=3):
        self.n = n
        self.data_types = data_types.split(',')
        self.dic_list = [generate_random_dict(self.data_types) for i in range(n)]
        self.dist_cutoff = dist_cutoff
        self.neighbor_val = neighbor_val

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        data_x = self.dic_list[idx]
        return dict2dgl(data_x, self.dist_cutoff, self.neighbor_val)


class Random_dict_Dataset(Dataset):

    def __init__(self, n, data_types='one_hot,position,features,lddt,cad,sh'):
        self.n = n
        self.data_types = data_types.split(',')
        self.dic_list = [generate_random_dict(self.data_types) for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.dic_list[idx]


def task_loss(pred, target, use_mean=True):
    l1_loss = torch.sum(torch.abs(pred - target))
    l2_loss = torch.sum((pred - target) ** 2)
    if use_mean:
        l1_loss /= pred.shape[0]
        l2_loss /= pred.shape[0]
    return l1_loss, l2_loss


def task_corr(pred, target):
    vx = pred - torch.mean(pred)
    vy = target - torch.mean(target)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return corr
