import os
import warnings

warnings.simplefilter(action='ignore',category=FutureWarning)

import dgl
import numpy as np
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from experiments.qm9 import models  # as models

batch_size = 4
device = torch.device('cuda:0')
div = 2.0
fully_connected = False
head = 8
log_interval = 25
lr = 0.001
model_type = 'SE3Transformer'
name = 'qm9-homo'
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
save_dir = 'models'
seed = 1992
task = 'homo'

node_feature_size = 6
edge_feature_size = 4

class RandomRotation(object):
    def __init__(self):
        pass
    def __call__(self,x):
        M = np.random.randn(3,3)
        Q,__ = np.linalg.qr(M)
        return x @ Q


def collate(samples):
    graphs,y = map(list,zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph,torch.tensor(y)

# Choose model
model = models.__dict__.get(model_type)(num_layers,
                                         node_feature_size,
                                         num_channels,
                                         num_nlayers=num_nlayers,
                                         num_degrees=num_degrees,
                                         edge_dim=edge_feature_size,
                                         div=div,
                                         pooling=pooling,
                                         n_heads=head)

model.to(device)
# Optimizer settings
optimizer = optim.Adam(model.parameters(),lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,num_epochs,eta_min=1e-4)


# Loss function
def task_loss(pred,target,use_mean=True):
    l1_loss = torch.sum(torch.abs(pred - target))
    l2_loss = torch.sum((pred - target) ** 2)
    if use_mean:
        l1_loss /= pred.shape[0]
        l2_loss /= pred.shape[0]
    return l1_loss,l2_loss

# Save path
save_path = os.path.join(save_dir,name + '.pt')

# Run training
print('Begin training')

class GDataset(Dataset):
    def __init__(self,num_nodes=20, num_edges=10,node_feature_size = 6, edge_feature_size=4,
                 size=300,dtype = np.float32):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.size = size
        self.dtype = dtype
    def __len__(self):
        return self.size
    def __getitem__(self,idx):
        y = np.array([np.random.random(1).astype(self.dtype)])
        # Create graph
        # G = dgl.DGLGraph((src,dst))
        G = dgl.rand_graph(self.num_nodes,self.num_edges)
        # Add node features to graph
        G.ndata['x'] = torch.tensor(np.random.random((self.num_nodes,3)).astype(self.dtype))  # [num_atoms,3]
        G.ndata['f'] = torch.tensor(np.random.random((self.num_nodes,node_feature_size,1)).astype(self.dtype)) # [num_atoms,6,1]
        # Add edge features to graph
        G.edata['d'] = torch.tensor(np.random.random((self.num_edges,3)).astype(self.dtype))  # [num_atoms,3]
        G.edata['w'] = torch.tensor(np.random.random((self.num_edges,edge_feature_size)).astype(self.dtype))  # [num_atoms,4]
        return G,y

test_loader = DataLoader(GDataset(),
                         batch_size=batch_size,
                         shuffle=False,
                         collate_fn=collate,
                         num_workers=num_workers)


num_iters = len(test_loader)
for epoch in range(num_epochs):
    torch.save(model.state_dict(),save_path)
    print(f"Saved: {save_path}")
    model.train()
    for i,(g,y) in enumerate(test_loader):
        print(i)
        g = g.to(device)
        y = y.to(device)
        print(g)
        pred = model(g)
        l1_loss,__ = task_loss(pred,y)
        l1_loss.backward()
        optimizer.step()
        scheduler.step(epoch + i / num_iters)
