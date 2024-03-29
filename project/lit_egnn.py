import os

import pytorch_lightning as pl
import torch
import wandb
from einops import rearrange
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.nn import ModuleList
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from project.datasets.RG.rg_dgl_data_module import RGDGLDataModule
from project.utils.metrics import L1Loss, L2Loss
from project.utils.modules import EnGraphConv
from project.utils.utils import collect_args, process_args, construct_wandb_pl_logger


class LitEGNN(pl.LightningModule):
    """An E(n)-equivariant GNN."""

    def __init__(self, node_feat: int = 512, pos_feat: int = 3, coord_feat: int = 16, edge_feat: int = 0,
                 fourier_feat: int = 0, num_nearest_neighbors: int = 3, num_classes: int = 2, num_layers: int = 4,
                 lr: float = 1e-3, num_epochs: int = 5):
        """Initialize all the parameters for a LitEGNN."""
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
        self.build_gnn_model()

        # Declare loss function(s) for training, validation, and testing
        self.L1Loss = L1Loss(1, 0, 'homo')
        self.L2Loss = L2Loss()

    def build_gnn_model(self):
        """Define the layers of a LitEGNN."""
        # Marshal all equivariant layers
        self.conv_block = ModuleList([EnGraphConv(node_feat=self.node_feat, edge_feat=self.edge_feat,
                                                  coord_feat=self.coord_feat, fourier_feat=self.fourier_feat,
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
        y = graph_and_labels[1]

        # Make a forward pass through the network
        h, x = self(h, x)

        # Calculate the loss
        l1_loss, rescaled_l1_loss = self.L1Loss(h, y)
        l2_loss = self.L2Loss(h, y)

        # Log training metrics
        self.log('train_l1_loss', l1_loss, sync_dist=True)
        self.log('train_rescaled_l1_loss', rescaled_l1_loss, sync_dist=True)
        self.log('train_l2_loss', l2_loss, sync_dist=True)

        return l1_loss

    def training_epoch_end(self, outs):
        """Lightning calls this at the end of every training epoch."""
        self.L1Loss.reset()
        self.L2Loss.reset()

    def validation_step(self, graph_and_labels, batch_idx):
        """Lightning calls this inside the validation loop."""
        h = rearrange(graph_and_labels[0].ndata['f'], 'n d () -> () n d')
        x = torch.randn(1, h.shape[1], 3).to(self.device)
        y = graph_and_labels[1]

        # Make a forward pass through the network
        h, x = self(h, x)

        # Calculate the loss
        l1_loss, rescaled_l1_loss = self.L1Loss(h, y)
        l2_loss = self.L2Loss(h, y)

        # Log training metrics
        self.log('val_l1_loss', l1_loss, sync_dist=True)
        self.log('val_rescaled_l1_loss', rescaled_l1_loss, sync_dist=True)
        self.log('val_l2_loss', l2_loss, sync_dist=True)

        return rescaled_l1_loss

    def validation_epoch_end(self, outs):
        """Lightning calls this at the end of every validation epoch."""
        self.L1Loss.reset()
        self.L2Loss.reset()

    def test_step(self, graph_and_labels, batch_idx):
        """Lightning calls this inside the testing loop."""
        h = rearrange(graph_and_labels[0].ndata['f'], 'n d () -> () n d')
        x = torch.randn(1, h.shape[1], 3).to(self.device)
        y = graph_and_labels[1]

        # Make a forward pass through the network
        h, x = self(h, x)

        # Calculate the loss
        l1_loss, rescaled_l1_loss = self.L1Loss(h, y)
        l2_loss = self.L2Loss(h, y)

        # Log training metrics
        self.log('test_l1_loss', l1_loss, sync_dist=True)
        self.log('test_rescaled_l1_loss', rescaled_l1_loss, sync_dist=True)
        self.log('test_l2_loss', l2_loss, sync_dist=True)

        return rescaled_l1_loss

    def test_epoch_end(self, outs):
        """Lightning calls this at the end of every testing epoch."""
        self.L1Loss.reset()
        self.L2Loss.reset()

    # ---------------------
    # Training Setup
    # ---------------------
    def configure_optimizers(self):
        """Called to configure the trainer's optimizer(s)."""
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer, self.num_epochs, eta_min=1e-4)
        metric_to_track = 'val_rescaled_l1_loss'
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': metric_to_track
        }


def cli_main():
    # -----------
    # Arguments
    # -----------
    args = collect_args()
    process_args(args)

    # -----------
    # Data
    # -----------
    data_module = RGDGLDataModule(batch_size=args.batch_size, num_dataloader_workers=args.num_workers)
    data_module.prepare_data()
    data_module.setup()

    # -----------
    # Model
    # -----------
    lit_egnn = LitEGNN(
        node_feat=data_module.num_node_features,
        pos_feat=data_module.num_pos_features,
        coord_feat=data_module.num_coord_features,
        edge_feat=0,  # Placeholder
        fourier_feat=data_module.num_fourier_features,
        num_nearest_neighbors=args.num_nearest_neighbors,
        num_classes=data_module.rg_train.out_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        num_epochs=args.num_epochs)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.max_epochs = args.num_epochs

    # Initialize logger
    args.experiment_name = f'EGNN-l{args.num_layers}-c{args.num_channels}' \
        if not args.experiment_name \
        else args.experiment_name

    # Log everything to Weights and Biases (WandB)
    logger = construct_wandb_pl_logger(args)

    # Assign specified logger (e.g. WandB) to Trainer instance
    trainer.logger = logger

    # -----------
    # Checkpoint
    # -----------
    # Resume from checkpoint if path to a valid one is provided
    args.ckpt_name = args.ckpt_name \
        if args.ckpt_name is not None \
        else 'LitEGNN-{epoch:02d}-{val_rescaled_l1_loss:.2f}.ckpt'
    checkpoint_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    trainer.resume_from_checkpoint = checkpoint_path if os.path.exists(checkpoint_path) else None

    # -----------
    # Training
    # -----------
    # Create and use callbacks
    early_stop_callback = EarlyStopping(monitor='val_rescaled_l1_loss', mode='min', min_delta=0.00, patience=3)
    checkpoint_callback = ModelCheckpoint(monitor='val_rescaled_l1_loss', save_top_k=3, dirpath=args.ckpt_dir,
                                          filename='LitEGNN-{epoch:02d}-{val_rescaled_l1_loss:.2f}')
    lr_callback = LearningRateMonitor(logging_interval='epoch')  # Use with a learning rate scheduler
    trainer.callbacks = [early_stop_callback, checkpoint_callback, lr_callback]

    # Train with the provided model and data module
    trainer.fit(lit_egnn, datamodule=data_module)

    # -----------
    # Testing
    # -----------
    trainer.test()


if __name__ == '__main__':
    cli_main()
