import dgl
import torch
from dgl.data import PPIDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader


class PPIDGLDataModule(LightningDataModule):
    """PPI data module for DGL with PyTorch."""

    # Dataset partition instantiations
    ppi_graph_dataset_train = None
    ppi_graph_dataset_val = None
    ppi_graph_dataset_test = None

    def __init__(self, batch_size=1, num_dataloader_workers=1):
        super().__init__()

        # Dataset meta-parameters
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers

    @property
    def num_node_features(self) -> int:
        return 50

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

    @property
    def num_classes(self) -> int:
        return 121

    def prepare_data(self):
        # Download the full dataset - called only on 1 GPU
        PPIDataset().download()

    def setup(self, stage=None):
        # Assign training/validation/testing data set for use in DataLoaders - called on every GPU
        self.ppi_graph_dataset_train = PPIDataset(mode='train')
        self.ppi_graph_dataset_val = PPIDataset(mode='valid')
        self.ppi_graph_dataset_test = PPIDataset(mode='test')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.ppi_graph_dataset_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_dataloader_workers, collate_fn=self.collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.ppi_graph_dataset_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=self.collate_fn)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.ppi_graph_dataset_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(graphs):
        """Group DGLGraphs into graph batches of a specified size."""
        # Curate graph batches
        batched_graphs = dgl.batch(graphs)
        # Curate label batches
        labels = torch.cat([graph.ndata['label'] for graph in graphs])
        return batched_graphs, labels
