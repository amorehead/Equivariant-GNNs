from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from project.datasets.RG.rg_dgl_dataset import RGDGLDataset
from project.utils.utils import collate


class RGDGLDataModule(LightningDataModule):
    """Random graph data module for DGL with PyTorch."""

    def __init__(self, node_feature_size=6, edge_feature_size=4, batch_size=4, num_dataloader_workers=4, seed=42):
        super().__init__()

        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.seed = seed

        # Dataset partition instantiations
        self.rg_train = None
        self.rg_val = None
        self.rg_test = None

    @property
    def num_atom_features(self) -> int:
        return self.rg_train.num_atom_features

    @property
    def num_bonds(self) -> int:
        return self.rg_train.num_bonds

    def prepare_data(self):
        # Download the full dataset - called only on 1 GPU
        self.rg_train = RGDGLDataset(mode='train',
                                     node_feature_size=self.node_feature_size,
                                     edge_feature_size=self.edge_feature_size)

    def setup(self, stage: Optional[str] = None):
        # Assign training/validation/testing data set for use in DataLoaders - called on every GPU
        self.rg_train = RGDGLDataset(mode='train',
                                     node_feature_size=self.node_feature_size,
                                     edge_feature_size=self.edge_feature_size)
        self.rg_val = RGDGLDataset(mode='val',
                                   node_feature_size=self.node_feature_size,
                                   edge_feature_size=self.edge_feature_size)
        self.rg_test = RGDGLDataset(mode='test',
                                    node_feature_size=self.node_feature_size,
                                    edge_feature_size=self.edge_feature_size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.rg_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_dataloader_workers, collate_fn=collate)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.rg_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=collate)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.rg_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=collate)
