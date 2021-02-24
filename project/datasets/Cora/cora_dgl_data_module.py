from typing import Optional

import dgl
import torch
from dgl.data import CoraFullDataset, CoraBinary
from numpy import random
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from project.utils.utils import collate


class CoraDGLDataModule(LightningDataModule):
    """Cora data module for DGL with PyTorch."""

    def __init__(self, batch_size=4, num_dataloader_workers=4, seed=42):
        super().__init__()

        # Dataset meta-parameters
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.seed = seed

        # Dataset partition instantiations
        self.cora_binary = None
        self.cora_binary_train = None
        self.cora_binary_val = None
        self.cora_binary_test = None

    @property
    def num_node_features(self) -> int:
        return 1

    @property
    def num_edge_features(self) -> int:
        return 1

    def prepare_data(self):
        # Download the full dataset - called only on 1 GPU
        self.cora_binary = CoraBinary()
        self.cora_binary.download()
        self.cora_binary = self.convert_cora_data_to_set_format(self.cora_binary)

    @staticmethod
    def convert_cora_data_to_set_format(cora_binary_dataset):
        reformatted_graphs = []
        for i, graph in enumerate(cora_binary_dataset.graphs):
            edges = graph.edges.__call__()
            reformatted_graph = dgl.graph((edges[0].data.tolist(), edges[1].data.tolist()))
            reformatted_graph.ndata['x'] = torch.tensor([[0.0, 0.0, 0.0] for _ in range(reformatted_graph.num_nodes())])  # [num_nodes,3]
            reformatted_graph.ndata['f'] = torch.tensor([0.0 for _ in range(reformatted_graph.num_nodes())])  # [num_nodes,node_feature_size]
            reformatted_graph.ndata['y'] = torch.from_numpy(cora_binary_dataset.labels[i])  # [num_nodes,1]
            reformatted_graph.edata['d'] = torch.tensor([[0.0, 0.0, 0.0] for _ in range(reformatted_graph.num_edges())])  # [num_edges,3]
            reformatted_graph.edata['w'] = torch.tensor([0.0 for _ in range(reformatted_graph.num_edges())])  # [num_nodes,edge_feature_size]
            reformatted_graphs.append(reformatted_graph)
        cora_binary_dataset.graphs = reformatted_graphs
        return cora_binary_dataset

    @staticmethod
    def collate_graphs_with_embedded_labels(graphs):
        nested_graphs = [tupl[0] for tupl in graphs]
        batched_graph = dgl.batch(nested_graphs)
        return batched_graph

    def setup(self, stage: Optional[str] = None):
        # Assign training/validation/testing data set for use in DataLoaders - called on every GPU
        self.cora_binary_train, self.cora_binary_val, self.cora_binary_test = \
            dgl.data.utils.split_dataset(self.cora_binary, frac_list=None, shuffle=False, random_state=None)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.cora_binary_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_dataloader_workers, collate_fn=self.collate_graphs_with_embedded_labels)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.cora_binary_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=self.collate_graphs_with_embedded_labels)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.cora_binary_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=self.collate_graphs_with_embedded_labels)
