from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from project.datasets.QM9.qm9_dgl_dataset import QM9DGLDataset
from project.utils.utils import RandomRotation, collate


class QM9DGLDataModule(LightningDataModule):
    """QM9 data module for DGL with PyTorch."""

    def __init__(self, data_dir: str, task: str, batch_size=32, num_dataloader_workers=1, seed=42):
        super().__init__()

        self.data_dir = data_dir
        self.task = task
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.seed = seed

        # Dataset partition instantiations
        self.qm9_train = None
        self.qm9_val = None
        self.qm9_test = None

    @property
    def num_atom_features(self) -> int:
        return self.qm9_train.num_atom_features

    @property
    def num_bonds(self) -> int:
        return self.qm9_train.num_bonds

    def prepare_data(self):
        # Download the full dataset - called only on 1 GPU
        QM9DGLDataset(self.data_dir, self.task, mode='train', transform=RandomRotation())

    def setup(self, stage: Optional[str] = None):
        # Assign training/validation/testing data set for use in DataLoaders - called on every GPU
        self.qm9_train = QM9DGLDataset(self.data_dir, self.task, mode='train', transform=RandomRotation())
        self.qm9_val = QM9DGLDataset(self.data_dir, self.task, mode='val')
        self.qm9_test = QM9DGLDataset(self.data_dir, self.task, mode='test')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.qm9_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_dataloader_workers, collate_fn=collate)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.qm9_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=collate)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.qm9_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=collate)
