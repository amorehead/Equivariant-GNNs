import os

import pytorch_lightning as pl
import torch
import wandb
from dgl.nn.pytorch import GraphConv
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch import softmax
from torch.nn import Embedding, CrossEntropyLoss
from torch.optim import Adam
from torchmetrics import Accuracy

from project.datasets.KarateClub.karate_club_dgl_data_module import KarateClubDGLDataModule
from project.utils.utils import collect_args, process_args, construct_wandb_pl_logger


class LitGraphSAGE(pl.LightningModule):
    """A GraphSAGE-based GNN."""

    def __init__(self, node_feat: int = 5, hidden_dim: int = 5, num_classes: int = 2,
                 num_hidden_layers: int = 0, lr: float = 0.01, num_epochs: int = 50):
        """Initialize all the parameters for a LitGraphSAGE GNN."""
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

        # Establish dataset variables
        self.embed = Embedding(34, 5)  # 34 nodes with embedding dim equal to 5
        self.labeled_nodes = torch.tensor([0, 33])  # Only the instructor and the president nodes are labeled
        self.labels = torch.tensor([0, 1])  # Their labels are different

        # Declare loss function for training, validation, and testing
        self.ce = CrossEntropyLoss()
        self.train_acc = Accuracy()

    def build_gnn_model(self):
        """Define the layers of a LitGraphSAGE GNN."""
        # Marshal all GNN layers
        self.conv1 = GraphConv(self.node_feat, self.hidden_dim)
        self.conv2 = GraphConv(self.hidden_dim, self.num_classes)

    # ---------------------
    # Training
    # ---------------------
    def gnn_forward(self, graph, feats):
        """Make a forward pass through the entire network."""
        logits = self.conv1(graph, feats)
        logits = torch.relu(logits)
        logits = self.conv2(graph, logits)
        return logits

    def forward(self, graph, feats):
        """Make a forward pass through the entire network."""
        # Forward propagate with both GNNs
        logits = self.gnn_forward(graph, feats)

        # Return network prediction
        return logits

    def training_step(self, batch, batch_idx):
        """Lightning calls this inside the training loop."""
        # Make a forward pass through the network for an entire batch of training graph pairs
        graph = batch
        graph.ndata['feat'] = self.embed.weight
        logits = self(graph, self.embed.weight)
        self.labels = self.labels.to(self.device)

        # Calculate the batch loss
        class_probs = softmax(logits, 1).to(self.device)
        loss = self.ce(class_probs[self.labeled_nodes], self.labels)  # Calculate CrossEntropyLoss of a single batch
        self.train_acc(class_probs[self.labeled_nodes], self.labels)  # Calculate Accuracy of a single batch

        # Log training step metric(s)
        self.log('train_ce', loss, on_step=True, sync_dist=True)
        self.log('train_acc', self.train_acc, on_step=True, sync_dist=True)

        return {'loss': loss}

    def training_epoch_end(self, outs):
        """Lightning calls this at the end of every training epoch."""
        self.log('train_acc', self.train_acc.compute(), sync_dist=True)  # Log Accuracy of an epoch
        self.train_acc.reset()

    # ---------------------
    # Training Setup
    # ---------------------
    def configure_optimizers(self):
        """Called to configure the trainer's optimizer(s)."""
        optimizer = Adam(self.parameters(), lr=self.lr)
        metric_to_track = 'train_ce'
        return {
            'optimizer': optimizer,
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
    karate_club_data_module = KarateClubDGLDataModule(batch_size=1, num_dataloader_workers=args.num_workers)
    karate_club_data_module.prepare_data()
    karate_club_data_module.setup()

    # ------------
    # Model
    # ------------
    lit_gs = LitGraphSAGE(
        node_feat=karate_club_data_module.num_node_features,
        hidden_dim=args.num_channels,
        num_classes=karate_club_data_module.karate_club_dataset.num_classes,
        num_hidden_layers=args.num_layers,
        lr=args.lr,
        num_epochs=args.num_epochs)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.max_epochs = args.num_epochs

    # Initialize logger
    args.experiment_name = f'LitGraphSAGE-l{args.num_layers}-h{args.num_channels}-b{args.batch_size}' \
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
    args.ckpt_name = args.ckpt_name if args.ckpt_name is not None else 'LitGraphSAGE-{epoch:02d}-{train_ce:.2f}.ckpt'
    checkpoint_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    trainer.resume_from_checkpoint = checkpoint_path if os.path.exists(checkpoint_path) else None

    # -----------
    # Training
    # -----------
    # Create and use callbacks
    early_stop_callback = EarlyStopping(monitor='train_ce', mode='min', min_delta=0.00, patience=10)
    checkpoint_callback = ModelCheckpoint(monitor='train_ce', save_top_k=3, dirpath=args.ckpt_dir,
                                          filename='LitGraphSAGE-{epoch:02d}-{train_ce:.2f}')
    lr_callback = LearningRateMonitor(logging_interval='epoch')  # Use with a learning rate scheduler
    trainer.callbacks = [early_stop_callback, checkpoint_callback, lr_callback]

    # Train with the provided model and data module
    trainer.fit(lit_gs, datamodule=karate_club_data_module)

    # -----------
    # Testing
    # -----------
    trainer.test()


if __name__ == '__main__':
    cli_main()
