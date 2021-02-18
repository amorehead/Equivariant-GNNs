import os
import warnings

import dgl
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from project.datasets.RG.rg_dgl_dataset import RGDGLDataset
from project.utils.fibers import Fiber
from project.utils.modules import GConvSE3, GNormSE3, GMaxPooling, get_basis_and_r, GSE3Res, GAvgPooling

warnings.simplefilter(action='ignore', category=FutureWarning)

batch_size = 4
device = torch.device('cuda:0')
div = 2.0
fully_connected = False
head = 8
log_interval = 25
lr = 0.001
model_type = 'SE3Transformer'
save_dir = 'test_model.pt'
num_channels = 32
num_degrees = 4
num_epochs = 1
num_layers = 7
num_nlayers = 0
num_workers = 4
pooling = 'max'
print_interval = 250
profile = False
restore = None
save_dir = '~/test_model.pt'
seed = 1992
task = 'homo'

node_feature_size = 6
edge_feature_size = 4


class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        return x @ Q


def collate(samples):
    graphs, y = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(y)


class TFN(nn.Module):
    """SE(3) equivariant GCN"""

    def __init__(self, num_layers: int, atom_feature_size: int,
                 num_channels: int, num_nlayers: int = 1, num_degrees: int = 4,
                 edge_dim: int = 4, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.num_channels_out = num_channels * num_degrees
        self.edge_dim = edge_dim

        self.fibers = {'in': Fiber(1, atom_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, self.num_channels_out)}

        blocks = self._build_gcn(self.fibers, 1)
        self.block0, self.block1, self.block2 = blocks
        print(self.block0)
        print(self.block1)
        print(self.block2)

    def _build_gcn(self, fibers, out_dim):
        block0 = []
        fin = fibers['in']
        for i in range(self.num_layers - 1):
            block0.append(GConvSE3(fin, fibers['mid'], self_interaction=True, edge_dim=self.edge_dim))
            block0.append(GNormSE3(fibers['mid'], num_layers=self.num_nlayers))
            fin = fibers['mid']
        block0.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        block1 = [GMaxPooling()]

        block2 = []
        block2.append(nn.Linear(self.num_channels_out, self.num_channels_out))
        block2.append(nn.ReLU(inplace=True))
        block2.append(nn.Linear(self.num_channels_out, out_dim))

        return nn.ModuleList(block0), nn.ModuleList(block1), nn.ModuleList(block2)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees - 1)

        # encoder (equivariant layers)
        h = {'0': G.ndata['f']}
        for layer in self.block0:
            h = layer(h, G=G, r=r, basis=basis)

        for layer in self.block1:
            h = layer(h, G)

        for layer in self.block2:
            h = layer(h)

        return h


class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""

    def __init__(self, num_layers: int, atom_feature_size: int,
                 num_channels: int, num_nlayers: int = 1, num_degrees: int = 4,
                 edge_dim: int = 4, div: float = 4, pooling: str = 'avg', n_heads: int = 1, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.pooling = pooling
        self.n_heads = n_heads

        self.fibers = {'in': Fiber(1, atom_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, num_degrees * self.num_channels)}

        blocks = self._build_gcn(self.fibers, 1)
        self.Gblock, self.FCblock = blocks
        print(self.Gblock)
        print(self.FCblock)

    def _build_gcn(self, fibers, out_dim):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim,
                                  div=self.div, n_heads=self.n_heads))
            Gblock.append(GNormSE3(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        # Pooling
        if self.pooling == 'avg':
            Gblock.append(GAvgPooling())
        elif self.pooling == 'max':
            Gblock.append(GMaxPooling())

        # FC layers
        FCblock = []
        FCblock.append(nn.Linear(self.fibers['out'].n_features, self.fibers['out'].n_features))
        FCblock.append(nn.ReLU(inplace=True))
        FCblock.append(nn.Linear(self.fibers['out'].n_features, out_dim))

        return nn.ModuleList(Gblock), nn.ModuleList(FCblock)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees - 1)

        # encoder (equivariant layers)
        h = {'0': G.ndata['f']}
        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)

        for layer in self.FCblock:
            h = layer(h)

        return h


def task_loss(pred, target, use_mean=True):
    l1_loss = torch.sum(torch.abs(pred - target))
    l2_loss = torch.sum((pred - target) ** 2)
    if use_mean:
        l1_loss /= pred.shape[0]
        l2_loss /= pred.shape[0]
    return l1_loss, l2_loss


# Choose model
model = SE3Transformer(num_layers, node_feature_size,
                       num_channels,
                       num_nlayers=num_nlayers,
                       num_degrees=num_degrees,
                       edge_dim=edge_feature_size,
                       div=div,
                       pooling=pooling,
                       n_heads=head)

model.to(device)
# Optimizer settings
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, num_epochs, eta_min=1e-4)

# Save path
save_path = os.path.join(save_dir)

# Run training
print('Begin training')
test_loader = DataLoader(RGDGLDataset(),
                         batch_size=batch_size,
                         shuffle=False,
                         collate_fn=collate,
                         num_workers=num_workers)

num_iters = len(test_loader)
for epoch in range(num_epochs):
    torch.save(model.state_dict(), save_path)
    print(f"Saved: {save_path}")
    model.train()
    for i, (g, y) in enumerate(test_loader):
        print(i)
        g = g.to(device)
        y = y.to(device)
        print(g)
        pred = model(g)
        l1_loss, __ = task_loss(pred, y)
        l1_loss.backward()
        optimizer.step()
        scheduler.step(epoch + i / num_iters)
