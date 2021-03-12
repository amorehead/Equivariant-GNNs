import dgl
import numpy as np
import torch
from equivariant_attention.fibers import Fiber
from equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res
from se3_transformer_pytorch import SE3Transformer
from se3_transformer_pytorch.irr_repr import rot
from torch import nn


class SE3Transformer_original(nn.Module):
    def __init__(self, num_layers: int, atom_feature_size: int,
                 num_channels: int, num_nlayers: int = 1, num_degrees: int = 4,
                 edge_dim: int = 4, div: float = 4, n_heads: int = 1, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.n_heads = n_heads
        self.fibers = {'in': Fiber(1, atom_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, num_degrees * self.num_channels)}
        self.Gblock = self._build_gcn(self.fibers, 1)

    def _build_gcn(self, fibers, out_dim):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim,
                                  div=self.div, n_heads=self.n_heads))
            Gblock.append(GNormSE3(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'],
                               self_interaction=True, edge_dim=self.edge_dim))
        return nn.ModuleList(Gblock)

    def forward(self, G):
        basis, r = get_basis_and_r(G, self.num_degrees - 1)
        # encoder (equivariant layers)
        h = G.ndata['f']
        return h


model1 = SE3Transformer(
    dim=16,
    depth=2,
    attend_self=True,
    num_degrees=2,
    output_degrees=1,
    fourier_encode_dist=True,
)
model_original = SE3Transformer_original(num_layers=3,
                                         atom_feature_size=16,
                                         num_channels=16,
                                         num_degrees=2,
                                         edge_dim=2)
device = 'cuda:0'
model_original.to(device)
model1.to(device)
feats = torch.randn(1, 32, 16).to(device)
coors = torch.randn(1, 32, 3).to(device)
mask = torch.ones(1, 32).bool().to(device)
R = rot(15, 0, 45).to(device)
edge_f = torch.randn(1, 32 * 32, 3).to(device)
import time

src_ids = np.repeat(range(32), 32)
dst_ids = np.tile(range(32), 32)
g = dgl.graph((src_ids, dst_ids), idtype=torch.long, device=device)
g.ndata['x'] = coors[0, :, :]
g.ndata['f'] = feats[0, :, :]
g.edata['d'] = g.ndata['x'][g.edges()[0]] - g.ndata['x'][g.edges()[1]]
g.edata['w'] = edge_f[0, :, :]
t1 = time.time()
for i in range(100):
    out1 = model_original(g)
t2 = time.time()
g.ndata['x'] = g.ndata['x'] @ R
out2 = model_original(g)
diff = (out1 - out2).max()
print(str(diff) + ' takes ' + str(t2 - t1))

edge_f = edge_f.view(32, 32, 3)
t1 = time.time()
for i in range(100):
    out1 = model1(feats, coors, mask, return_type=1)
t2 = time.time()
out2 = model1(feats, coors @ R, mask, return_type=1)
diff = (out1 - out2).max()
print(str(diff) + ' takes ' + str(t2 - t1))
