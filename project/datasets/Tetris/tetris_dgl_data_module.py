from typing import Optional

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from project.datasets.Tetris.tetris_dgl_dataset import TetrisDGLDataset
from project.utils.utils import collate


class TetrisDGLDataModule(LightningDataModule):
    """Tetris data module for DGL with PyTorch."""

    # Dataset partition instantiations
    tetris_train = None
    tetris_val = None
    tetris_test = None

    def __init__(self, transform=None, fill=0, dtype=np.float32, batch_size=4, num_dataloader_workers=4):
        super().__init__()

        # Dataset parameters
        self.transform = transform
        self.fill = fill
        self.dtype = dtype

        # Dataset meta-parameters
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers

    @property
    def num_node_features(self) -> int:
        return 1

    @property
    def num_pos_features(self) -> int:
        return 3

    @property
    def num_coord_features(self) -> int:
        return 3

    @property
    def num_edge_features(self) -> int:
        return 0

    @property
    def num_fourier_features(self) -> int:
        return 0

    def prepare_data(self):
        # Download the full dataset - called only on 1 GPU
        self.tetris_train = TetrisDGLDataset(transform=None, fill=self.fill, dtype=np.float32)

    def setup(self, stage: Optional[str] = None):
        # Assign training/validation/testing data set for use in DataLoaders - called on every GPU
        self.tetris_train = TetrisDGLDataset(transform=None, fill=self.fill, dtype=np.float32)
        self.tetris_val = TetrisDGLDataset(transform=None, fill=self.fill, dtype=np.float32)
        self.tetris_test = TetrisDGLDataset(transform=None, fill=self.fill, dtype=np.float32)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.tetris_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_dataloader_workers, collate_fn=collate)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.tetris_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=collate)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.tetris_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=collate)
