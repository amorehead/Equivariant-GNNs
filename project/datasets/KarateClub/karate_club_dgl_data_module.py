from dgl.data import KarateClubDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader


class KarateClubDGLDataModule(LightningDataModule):
    """Karate club data module for DGL with PyTorch."""

    # Dataset partition instantiations
    karate_club_dataset = None

    def __init__(self, batch_size=1, num_dataloader_workers=1):
        super().__init__()

        # Dataset meta-parameters
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers

    @property
    def num_node_features(self) -> int:
        return 5

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
        self.karate_club_dataset = KarateClubDataset()
        self.karate_club_dataset.download()

    def collate_fn(self, dataset):
        """A custom collate function for working with the DGL built-in KarateClubDataset."""
        graph = dataset[0]
        return graph

    def setup(self, stage=None):
        # Assign training/validation/testing data set for use in DataLoaders - called on every GPU
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.karate_club_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=self.collate_fn)
