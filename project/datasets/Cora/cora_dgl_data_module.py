from typing import Optional

import dgl
from dgl.data import CoraFullDataset, CoraGraphDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader


class CoraDGLDataModule(LightningDataModule):
    """Cora data module for DGL with PyTorch."""

    # Dataset partition instantiations
    cora_graph_dataset = None
    cora_graph_dataset_train = None
    cora_graph_dataset_val = None
    cora_graph_dataset_test = None

    def __init__(self, batch_size=1, num_dataloader_workers=1):
        super().__init__()

        # Dataset meta-parameters
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers

    @property
    def num_node_features(self) -> int:
        return 1433

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
        self.cora_graph_dataset = CoraGraphDataset()
        self.cora_graph_dataset.download()

    def collate_fn(self, samples):
        """A custom collate function for working with the DGL built-in CoraGraphDataset."""
        graph = samples[0]
        return graph

    def setup(self, stage: Optional[str] = None):
        # Assign training/validation/testing data set for use in DataLoaders - called on every GPU
        self.cora_graph_dataset_train, self.cora_graph_dataset_val, self.cora_graph_dataset_test = \
            dgl.data.utils.split_dataset(self.cora_graph_dataset, frac_list=None, shuffle=False, random_state=None)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.cora_graph_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=self.collate_fn)
