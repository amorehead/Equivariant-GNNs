import os

import pytorch_lightning as pl
import torch
from einops import rearrange
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.nn import ModuleList
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from project.datasets.RG.rg_dgl_data_module import RGDGLDataModule
from project.utils.modules import GConvEn
from project.utils.utils import collect_args, process_args


class LitEGNN(pl.LightningModule):
    """An E(n)-equivariant GNN."""

    def __init__(self, node_feat: int = 512, pos_feat: int = 3, coord_feat: int = 16, edge_feat: int = 0,
                 fourier_feat: int = 0, num_nearest_neighbors: int = 3, num_classes: int = 2, num_layers: int = 4,
                 lr: float = 1e-3, num_epochs: int = 5):
        """Initialize all the parameters for an EGNN."""
        super().__init__()
        self.save_hyperparameters()

        # Build the network
        self.node_feat = node_feat
        self.pos_feat = pos_feat
        self.coord_feat = coord_feat
        self.edge_feat = edge_feat
        self.fourier_feat = fourier_feat
        self.num_nearest_neighbors = num_nearest_neighbors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.lr = lr
        self.num_epochs = num_epochs

        # Assemble the layers of the network
        self.build_gcn_model()

        # Declare loss function(s) for training, validation, and testing
        self.cross_entropy = cross_entropy

    def build_gcn_model(self):
        """Define the layers of a single EGNN."""
        # Marshal all equivariant layers
        self.conv_block = ModuleList([GConvEn(self.num_layers, self.node_feat, self.pos_feat,
                                              self.edge_feat, self.coord_feat, self.fourier_feat,
                                              num_nearest_neighbors=self.num_nearest_neighbors)
                                      for _ in range(self.num_layers)])

    # ---------------------
    # Training
    # ---------------------
    def forward(self, h, x, e=None):
        """Make a forward pass through the entire network."""
        for layer in self.conv_block:
            h, x = layer(h, x, e)
        return h, x

    def training_step(self, graph_and_labels, batch_idx):
        """Lightning calls this inside the training loop."""
        h = rearrange(graph_and_labels[0].ndata['f'], 'n d () -> () n d')
        x = torch.randn(1, h.shape[1], 3).to(self.device)
        labels = graph_and_labels[1]

        # Make a forward pass through the network
        h, x = self.forward(h, x)
        # h, x = self.forward(h, x, e)

        # Calculate the loss
        cross_entropy = self.cross_entropy(h, labels)

        # Log training metrics
        self.log('cross_entropy', cross_entropy)

        return cross_entropy

    # ---------------------
    # Training Setup
    # ---------------------
    def configure_optimizers(self):
        """Called to configure the trainer's optimizer(s)."""
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer, self.num_epochs, eta_min=1e-4)
        metric_to_track = 'test_acc'
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': metric_to_track
        }

    def configure_callbacks(self):
        early_stop = EarlyStopping(monitor="test_acc", mode="max")
        checkpoint = ModelCheckpoint(monitor="test_acc", save_top_k=3)
        return [early_stop, checkpoint]


def cli_main():
    # -----------
    # Arguments
    # -----------
    args, unparsed_argv = collect_args()
    process_args(args, unparsed_argv)

    # Define HPC-specific properties in-file
    # args.accelerator = 'ddp'
    # args.distributed_backend = 'ddp'
    # args.plugins = 'ddp_sharded'
    args.gpus = 1
    args.precision = 16

    # -----------
    # Data
    # -----------
    data_module = RGDGLDataModule(batch_size=args.batch_size, num_dataloader_workers=args.num_workers)
    data_module.prepare_data()
    data_module.setup()

    # ------------
    # Checkpoint
    # ------------
    checkpoint_save_path = os.path.join(args.save_dir, f'{args.name}.pth')
    try:
        lit_egnn = LitEGNN.load_from_checkpoint(checkpoint_save_path)
        print(f'Resuming from checkpoint {checkpoint_save_path}\n')
    except:
        # -----------
        # Model
        # -----------
        print(f'Could not restore checkpoint {checkpoint_save_path}. Skipping...\n')
        lit_egnn = LitEGNN(
            node_feat=data_module.num_node_features,
            pos_feat=data_module.num_pos_features,
            coord_feat=data_module.num_coord_features,
            edge_feat=data_module.num_edge_features,
            fourier_feat=data_module.num_fourier_features,
            num_nearest_neighbors=args.num_nearest_neighbors,
            num_classes=data_module.rg_train.out_dim,
            num_layers=args.num_layers,
            lr=args.lr,
            num_epochs=args.num_epochs)

    # -----------
    # Training
    # -----------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.min_epochs = args.num_epochs

    # Logging all args to wandb
    # logger = construct_wandb_pl_logger(args)
    # trainer.logger = logger

    trainer.fit(lit_egnn, datamodule=data_module)

    # -----------
    # Testing
    # -----------
    # rg_test_results = trainer.test()
    # print(f'Model testing results on dataset: {rg_test_results}\n')

    # ------------
    # Finalizing
    # ------------
    print(f'Saving checkpoint {checkpoint_save_path}\n')
    trainer.save_checkpoint(checkpoint_save_path)


if __name__ == '__main__':
    cli_main()
