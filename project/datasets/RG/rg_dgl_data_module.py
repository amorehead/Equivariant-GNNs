from typing import Optional

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from project.datasets.RG.rg_dgl_dataset import RGDGLDataset
from project.utils.utils import collate, rand_rot


class RGDGLDataModule(LightningDataModule):
    """Random graph data module for DGL with PyTorch."""

    def __init__(self, n_lb=10, n_hb=20, e_lb=10, e_hb=15, node_feature_size=6, edge_feature_size=4, size=300,
                 out_dim=1, train_transform=None, test_transform=rand_rot, dtype=np.float32, batch_size=4,
                 num_dataloader_workers=4, seed=42):
        super().__init__()

        # Dataset parameters
        self.n_lb = n_lb
        self.n_hb = n_hb
        self.e_lb = e_lb
        self.e_hb = e_hb
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.size = size
        self.out_dim = out_dim
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.dtype = dtype

        # Dataset meta-parameters
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.seed = seed

        # Dataset partition instantiations
        self.rg_train = None
        self.rg_val = None
        self.rg_test = None

    def prepare_data(self):
        # Download the full dataset - called only on 1 GPU
        self.rg_train = RGDGLDataset(n_lb=self.n_lb, n_hb=self.n_hb, e_lb=self.e_lb, e_hb=self.e_hb,
                                     node_feature_size=self.node_feature_size, edge_feature_size=self.edge_feature_size,
                                     size=self.size, out_dim=self.out_dim, transform=self.train_transform,
                                     dtype=self.dtype)

    def setup(self, stage: Optional[str] = None):
        # Assign training/validation/testing data set for use in DataLoaders - called on every GPU
        self.rg_train = RGDGLDataset(n_lb=self.n_lb, n_hb=self.n_hb, e_lb=self.e_lb, e_hb=self.e_hb,
                                     node_feature_size=self.node_feature_size, edge_feature_size=self.edge_feature_size,
                                     size=self.size, out_dim=self.out_dim, transform=self.train_transform,
                                     dtype=self.dtype)
        self.rg_val = RGDGLDataset(n_lb=self.n_lb, n_hb=self.n_hb, e_lb=self.e_lb, e_hb=self.e_hb,
                                   node_feature_size=self.node_feature_size, edge_feature_size=self.edge_feature_size,
                                   size=self.size, out_dim=self.out_dim, transform=self.test_transform,
                                   dtype=self.dtype)
        self.rg_test = RGDGLDataset(n_lb=self.n_lb, n_hb=self.n_hb, e_lb=self.e_lb, e_hb=self.e_hb,
                                    node_feature_size=self.node_feature_size, edge_feature_size=self.edge_feature_size,
                                    size=self.size, out_dim=self.out_dim, transform=self.test_transform,
                                    dtype=self.dtype)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.rg_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_dataloader_workers, collate_fn=collate)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.rg_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=collate)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.rg_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=collate)
