import os

import pytorch_lightning as pl
from dgl.nn.pytorch import GATConv
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.nn import BCEWithLogitsLoss, ModuleList
from torch.nn.functional import elu
from torch.optim import Adam
from torchmetrics import F1

from project.datasets.PPI.ppi_dgl_data_module import PPIDGLDataModule
from project.utils.utils import collect_args, process_args, construct_wandb_pl_logger


class LitGAT(pl.LightningModule):
    """A GAT-based GNN."""

    def __init__(self, node_feat: int = 5, hidden_dim: int = 5, num_classes: int = 2,
                 num_hidden_layers: int = 0, lr: float = 0.01, num_epochs: int = 50):
        """Initialize all the parameters for a LitGAT GNN."""
        super().__init__()
        self.save_hyperparameters()

        # Build the network
        self.node_feat = node_feat
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_hidden_layers = num_hidden_layers
        self.lr = lr
        self.num_epochs = num_epochs

        # Assemble the layers of the network
        self.conv_block = self.build_gnn_model()

        # Declare loss function(s) for training, validation, and testing
        self.bce = BCEWithLogitsLoss(reduction='mean')
        self.train_f1 = F1(num_classes=self.num_classes)
        self.val_f1 = F1(num_classes=self.num_classes)
        self.test_f1 = F1(num_classes=self.num_classes)

    def build_gnn_model(self):
        """Define the layers of a LitGAT GNN."""
        # Marshal all GNN layers
        # Input projection (no residual)
        heads = [4, 4, 6]
        conv_block = [GATConv(in_feats=self.node_feat, out_feats=self.hidden_dim, num_heads=heads[0], activation=elu)]
        # Hidden layers
        for l in range(1, self.num_hidden_layers):
            # Due to multi-head, the in_dim = num_hidden * num_heads
            conv_block.append(
                GATConv(self.hidden_dim * heads[l - 1], self.hidden_dim, heads[l], residual=True, activation=elu))
        # Output projection
        conv_block.append(GATConv(
            self.hidden_dim * heads[-2], self.num_classes, heads[-1], residual=True))

        return ModuleList(conv_block)

    # ---------------------
    # Training
    # ---------------------
    def gnn_forward(self, graph, feats):
        """Make a forward pass through the entire network."""
        for i in range(self.num_hidden_layers):
            feats = self.conv_block[i](graph, feats).flatten(1)
        # Output projection
        logits = self.conv_block[-1](graph, feats).mean(1)
        return logits

    def forward(self, graph, feats):
        """Make a forward pass through the entire network."""
        # Forward propagate with both GNNs
        logits = self.gnn_forward(graph, feats)

        # Return network prediction
        return logits.squeeze()

    def training_step(self, batch, batch_idx):
        """Lightning calls this inside the training loop."""
        graphs, labels = batch

        # Make a forward pass through the network for an entire batch of training graph pairs
        logits = self(graphs, graphs.ndata['feat'])

        # Compute prediction
        preds = logits

        # Calculate the batch loss
        bce = self.bce(logits, labels)  # Calculate BCE of a single batch

        # Log training step metric(s)
        self.log('train_bce', bce, sync_dist=True)
        self.log('train_f1', self.train_f1(preds, labels), sync_dist=True)

        return {'loss': bce}

    def training_epoch_end(self, outputs):
        self.log('train_f1', self.train_f1.compute())
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        """Lightning calls this inside the validation loop."""
        graphs, labels = batch

        # Make a forward pass through the network for an entire batch of validation graph pairs
        logits = self(graphs, graphs.ndata['feat'])

        # Compute prediction
        preds = logits

        # Calculate the batch loss
        bce = self.bce(logits, labels)  # Calculate BCE of a single batch

        # Log validation step metric(s)
        self.log('val_bce', bce, sync_dist=True)
        self.log('val_f1', self.val_f1(preds, labels), sync_dist=True)

        return {'loss': bce}

    def validation_epoch_end(self, outputs):
        self.log('val_f1', self.val_f1.compute())
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        """Lightning calls this inside the testing loop."""
        graphs, labels = batch

        # Make a forward pass through the network for an entire batch of testing graph pairs
        logits = self(graphs, graphs.ndata['feat'])

        # Compute prediction
        preds = logits

        # Calculate the batch loss
        bce = self.bce(logits, labels)  # Calculate BCE of a single batch

        # Log testing step metric(s)
        self.log('test_bce', bce, sync_dist=True)
        self.log('test_f1', self.test_f1(preds, labels), sync_dist=True)

        return {'loss': bce}

    def test_epoch_end(self, outputs):
        self.log('test_f1', self.test_f1.compute())
        self.test_f1.reset()

    # ---------------------
    # Training Setup
    # ---------------------
    def configure_optimizers(self):
        """Called to configure the trainer's optimizer(s)."""
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer


def cli_main():
    # -----------
    # Arguments
    # -----------
    args = collect_args()
    process_args(args)

    # -----------
    # Data
    # -----------
    ppi_data_module = PPIDGLDataModule(batch_size=args.batch_size, num_dataloader_workers=args.num_workers)
    ppi_data_module.prepare_data()
    ppi_data_module.setup()

    # ------------
    # Model
    # ------------
    lit_gat = LitGAT(
        node_feat=ppi_data_module.num_node_features,
        hidden_dim=args.num_channels,
        num_classes=ppi_data_module.num_classes,
        num_hidden_layers=args.num_layers,
        lr=args.lr,
        num_epochs=args.num_epochs)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.max_epochs = args.num_epochs

    # Initialize logger
    args.experiment_name = f'LitGAT-l{args.num_layers}-h{args.num_channels}-b{args.batch_size}' \
        if not args.experiment_name \
        else args.experiment_name

    # Log everything to Weights and Biases (WandB)
    logger = construct_wandb_pl_logger(args)

    # Assign specified logger (e.g. WandB) to Trainer instance
    trainer.logger = logger

    # ------------
    # Checkpoint
    # ------------
    # Resume from checkpoint if path to a valid one is provided
    args.ckpt_name = args.ckpt_name if args.ckpt_name is not None else 'LitGAT-{epoch:02d}-{val_f1:.2f}.ckpt'
    checkpoint_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    trainer.resume_from_checkpoint = checkpoint_path if os.path.exists(checkpoint_path) else None

    # -----------
    # Training
    # -----------
    # Create and use callbacks
    early_stop_callback = EarlyStopping(monitor='val_f1', mode='max', min_delta=0.01, patience=10)
    checkpoint_callback = ModelCheckpoint(monitor='val_f1', save_top_k=3, dirpath=args.ckpt_dir,
                                          filename='LitGAT-{epoch:02d}-{val_f1:.2f}')
    lr_callback = LearningRateMonitor(logging_interval='epoch')  # Use with a learning rate scheduler
    trainer.callbacks = [early_stop_callback, checkpoint_callback, lr_callback]

    # Train with the provided model and data module
    trainer.fit(lit_gat, datamodule=ppi_data_module)

    # -----------
    # Testing
    # -----------
    trainer.test()


if __name__ == '__main__':
    cli_main()
