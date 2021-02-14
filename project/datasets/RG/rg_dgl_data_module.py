from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from project.datasets.RG.rg_dgl_dataset import RGDGLDataset
from project.utils.utils import RandomRotation, collate


class RGDGLDataModule(LightningDataModule):
    """Random graph data module for DGL with PyTorch."""

    def __init__(self, data_dir: str, task: str, batch_size=32, num_dataloader_workers=1, seed=42):
        super().__init__()

        self.data_dir = data_dir
        self.task = task
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
        RGDGLDataset(self.data_dir, self.task, mode='train', transform=RandomRotation())

    def setup(self, stage: Optional[str] = None):
        # Assign training/validation/testing data set for use in DataLoaders - called on every GPU
        self.rg_train = RGDGLDataset(self.data_dir, self.task, mode='train', transform=RandomRotation())
        self.rg_val = RGDGLDataset(self.data_dir, self.task, mode='valid')
        self.rg_test = RGDGLDataset(self.data_dir, self.task, mode='test')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.rg_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_dataloader_workers, collate_fn=collate)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.rg_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=collate)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.rg_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=collate)
