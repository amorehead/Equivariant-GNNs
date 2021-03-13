import time

import dgl
import numpy as np
import torch
from equivariant_attention.fibers import Fiber
from equivariant_attention.modules import GSE3Res, GNormSE3, GConvSE3, get_basis_and_r
from se3_transformer_pytorch import SE3Transformer
from se3_transformer_pytorch.irr_repr import rot
from torch.nn import Module, ModuleList

from project.lit_egnn import LitEGNN


# Instantiate different models to be compared
# original_se3_transformer = LitSET(num_layers=3,
#                                   atom_feature_size=16,
#                                   num_channels=16,
#                                   num_degrees=2,
#                                   edge_dim=2)
class OriginalSE3Transformer(Module):
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
        return ModuleList(Gblock)

    def forward(self, G):
        basis, r = get_basis_and_r(G, self.num_degrees - 1)
        # encoder (equivariant layers)
        h = {'0': G.ndata['f']}
        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)
        x = h['0'][..., -1]
        return x


original_se3_transformer = OriginalSE3Transformer(num_layers=3,
                                                  atom_feature_size=16,
                                                  num_channels=16,
                                                  num_degrees=2,
                                                  edge_dim=2)
new_se3_transformer = SE3Transformer(dim=16,
                                     depth=2,
                                     attend_self=True,
                                     num_degrees=2,
                                     output_degrees=1,
                                     fourier_encode_dist=True)
open_source_egnn = LitEGNN(node_feat=16,
                           pos_feat=3,
                           coord_feat=3,
                           edge_feat=3,
                           fourier_feat=0,
                           num_nearest_neighbors=0,
                           num_classes=1,
                           num_layers=3,
                           lr=1e-3,
                           num_epochs=5)

# Establish common variables
device = 'cuda:0'
original_se3_transformer.to(device)
new_se3_transformer.to(device)
open_source_egnn.to(device)

feats = torch.randn(1, 32, 16).to(device)
coors = torch.randn(1, 32, 3).to(device)
mask = torch.ones(1, 32).bool().to(device)
R = rot(15, 0, 45).to(device)
edge_f = torch.randn(1, 32 * 32, 3).to(device)

# Create mock graphs
src_ids = np.repeat(range(32), 32)
dst_ids = np.tile(range(32), 32)
g = dgl.graph((src_ids, dst_ids), idtype=torch.long, device=device)
g.ndata['x'] = coors[0, :, :]
g.ndata['f'] = feats[0, :, :].view(32, 16, 1)
g.edata['d'] = g.ndata['x'][g.edges()[0]] - g.ndata['x'][g.edges()[1]]
g.edata['w'] = edge_f[0, :, :]
output1 = None
num_steps = 10

# Test original SE(3)-Transformer (by Fabian Fuchs et al.)
t1 = time.time()
for i in range(num_steps):
    output1 = original_se3_transformer(g)

t2 = time.time()
g.ndata['x'] = g.ndata['x'] @ R
output2 = original_se3_transformer(g)
diff = (output1 - output2).max()
print(
    f'The original SE(3)-Transformer, with an equivariance error of {str(diff)}, takes {str(t2 - t1)} seconds to perform {num_steps} forward passes.')

# Test new SE(3)-Transformer (by lucidrains et al.)
edge_f = edge_f.view(32, 32, 3)
t1 = time.time()
for i in range(num_steps):
    output1 = new_se3_transformer(feats, coors, mask, return_type=1)

t2 = time.time()
output2 = new_se3_transformer(feats, coors @ R, mask, return_type=1)
diff = (output1 - output2).max()
print(
    f'The new SE(3)-Transformer, with an equivariance error of {str(diff)}, takes {str(t2 - t1)} seconds to perform {num_steps} forward passes.')

# Test the open-source EGNN (by lucidrains et al.)
edge_f = edge_f.view(32, 32, 3)
t1 = time.time()
for i in range(num_steps):
    output1 = open_source_egnn(feats, coors)
t2 = time.time()
output2 = new_se3_transformer(feats, coors @ R)
diff = (output1 - output2).max()
print(
    f'The open-source EGNN, with an equivariance error of {str(diff)}, takes {str(t2 - t1)} seconds to perform {num_steps} forward passes.')
