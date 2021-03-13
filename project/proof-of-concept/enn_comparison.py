import time

import dgl
import numpy as np
import torch
from se3_transformer_pytorch import SE3Transformer
from se3_transformer_pytorch.irr_repr import rot

from project.lit_egnn import LitEGNN
from project.lit_set import LitSET

# Instantiate different models to be compared
original_se3_transformer = LitSET(num_layers=3,
                                  atom_feature_size=16,
                                  num_channels=16,
                                  num_degrees=2,
                                  edge_dim=3)
new_se3_transformer = SE3Transformer(dim=16,
                                     depth=2,
                                     attend_self=True,
                                     num_degrees=2,
                                     output_degrees=1,
                                     fourier_encode_dist=True)
open_source_egnn = LitEGNN(node_feat=16,
                           pos_feat=3,
                           coord_feat=3,
                           edge_feat=0,
                           fourier_feat=0,
                           num_nearest_neighbors=0,
                           num_layers=3)

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
    output1, x = open_source_egnn(feats, coors)
t2 = time.time()
output2 = new_se3_transformer(feats, coors @ R)
diff = (output1 - output2).max()
print(
    f'The open-source EGNN, with an equivariance error of {str(diff)}, takes {str(t2 - t1)} seconds to perform {num_steps} forward passes.')
