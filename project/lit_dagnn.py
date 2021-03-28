import os

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch import relu
from torch.nn import Linear, ModuleList, Dropout, Module, BCEWithLogitsLoss, ReLU
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import AUROC, AveragePrecision, Accuracy

from project.datasets.KarateClub.karate_club_dgl_data_module import KarateClubDGLDataModule
from project.utils.modules import MLPLayer, DAGNNConv
from project.utils.utils import collect_args, process_args, construct_wandb_pl_logger


class LitDAGNN(pl.LightningModule):
    """A Deep Adaptive Graph Neural Network (DAGNN) for PyTorch Lightning."""

    def __init__(self, node_feat: int = 29, num_classes: int = 2, num_hidden_layers: int = 12, hidden_dim: int = 32,
                 bias=True, activation: Module = relu, lr: float = 1e-3, weight_decay: float = 5e-3,
                 num_epochs: int = 5, dropout_rate: float = 0.5):
        """Initialize all the parameters for a LitDAGNN."""
        super().__init__()
        self.save_hyperparameters()

        # Build the network
        self.node_feat = node_feat
        self.num_classes = num_classes
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.activation = activation
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate

        # Assemble the layers of the network
        self.build_gnn_model()
        self.build_ffn_model()

        # Declare loss function for training, validation, and testing
        self.bce = BCEWithLogitsLoss()
        self.val_acc = Accuracy()
        self.val_auroc = AUROC(pos_label=1)
        self.val_auprc = AveragePrecision(pos_label=1)
        self.test_acc = Accuracy()
        self.test_auroc = AUROC(pos_label=1)
        self.test_auprc = AveragePrecision(pos_label=1)

    def build_gnn_model(self):
        """Define the MLP layers of a LitDAGNN."""
        # Marshal all MLP layers
        self.mlp_block = ModuleList([MLPLayer(in_dim=self.node_feat, out_dim=self.hidden_dim, bias=self.bias,
                                              activation=self.activation, dropout=self.dropout_rate),
                                     MLPLayer(in_dim=self.hidden_dim, out_dim=self.node_feat, bias=self.bias,
                                              activation=None, dropout=self.dropout_rate)])
        self.dagnn = DAGNNConv(in_dim=self.node_feat, k=self.num_hidden_layers)

    def build_ffn_model(self):
        """Define the layers of the FFN."""
        # Marshal all FFN layers
        self.ff_block = ModuleList([Linear(self.node_feat, self.node_feat * 2),
                                    ReLU(),
                                    Dropout(p=self.dropout_rate),  # Randomly permute network structure
                                    Linear(self.node_feat * 2, self.num_classes - 1)])

    # ---------------------
    # Training
    # ---------------------
    def gnn_forward(self, graph):
        """Make a forward pass through the entire network."""
        feats = graph.ndata['f'].squeeze()
        for layer in self.mlp_block:
            feats = layer(feats)
        feats = self.dagnn(graph, feats)
        return feats

    def ffn_forward(self, graph1_feats, graph2_feats):
        """Make a forward pass through the FFN."""
        # Aggregate node representation batches and label batches
        logits = torch.cat((graph1_feats, graph2_feats))

        # Fully-connect node representation batch
        for layer in self.ff_block:
            logits = layer(logits)

        return logits

    def forward(self, graph1, graph2):
        """Make a forward pass through the entire network."""
        # Forward propagate with both GNNs - must add self loops to avoid exploding logit values
        graph1_feats = self.gnn_forward(graph1.add_self_loop())
        graph2_feats = self.gnn_forward(graph2.add_self_loop())

        # Merge latent representations with the FFN
        logits = self.ffn_forward(graph1_feats, graph2_feats)

        # Return network prediction
        return logits.squeeze()

    def training_step(self, batch, batch_idx):
        """Lightning calls this inside the training loop."""
        # Make a forward pass through the network for an entire batch of training graph pairs
        graphs1, graphs2, labels = batch[0], batch[1], batch[2]
        logits = self(graphs1, graphs2)

        # Calculate the batch loss
        bce = self.bce(logits, labels.float())  # Calculate BCE of a single batch

        # Log training step metric(s)
        self.log('train_bce', bce, sync_dist=True)

        return {'loss': bce}

    def validation_step(self, batch, batch_idx):
        """Lightning calls this inside the validation loop."""
        # Make a forward pass through the network for an entire batch of validation graph pairs
        graphs1, graphs2, labels = batch[0], batch[1], batch[2]
        logits = self(graphs1, graphs2)

        # Calculate the batch loss
        bce = self.bce(logits, labels.float())  # Calculate BCE of a single batch
        self.val_acc(logits.sigmoid(), labels)  # Calculate Accuracy of a single batch
        self.val_auroc(logits.sigmoid(), labels)  # Calculate AUROC of a single batch
        self.val_auprc(logits.sigmoid(), labels)  # Calculate AveragePrecision of a batch

        # Log validation step metric(s)
        self.log('val_bce', bce, sync_dist=True)

        return {'loss': bce}

    def validation_epoch_end(self, outs):
        """Lightning calls this at the end of every validation epoch."""
        self.log('val_acc', self.val_acc.compute(), sync_dist=True)  # Log Accuracy of an epoch
        self.log('val_auroc', self.val_auroc.compute(), sync_dist=True)  # Log AUROC of an epoch
        self.log('val_auprc', self.val_auprc.compute(), sync_dist=True)  # Log AveragePrecision of an epoch
        self.val_acc.reset()
        self.val_auroc.reset()
        self.val_auprc.reset()

    def test_step(self, batch, batch_idx):
        """Lightning calls this inside the testing loop."""
        # Make a forward pass through the network for an entire batch of test graph pairs
        graphs1, graphs2, labels = batch[0], batch[1], batch[2]
        logits = self(graphs1, graphs2)

        # Calculate the batch loss
        bce = self.bce(logits, labels.float())  # Calculate BCE of a single batch
        self.test_acc(logits.sigmoid(), labels)  # Calculate Accuracy of a single batch
        self.test_auroc(logits.sigmoid(), labels)  # Calculate AUROC of a batch
        self.test_auprc(logits.sigmoid(), labels)  # Calculate AveragePrecision of a batch

        # Log test step metric(s)
        self.log('test_bce', bce, sync_dist=True)

        return {'loss': bce}

    def test_epoch_end(self, outs):
        """Lightning calls this at the end of every test epoch."""
        self.log('test_acc', self.test_acc.compute(), sync_dist=True)  # Log Accuracy of an epoch
        self.log('test_auroc', self.test_auroc.compute(), sync_dist=True)  # Log AUROC of an epoch
        self.log('test_auprc', self.test_auprc.compute(), sync_dist=True)  # Log AveragePrecision of an epoch
        self.test_acc.reset()
        self.test_auroc.reset()
        self.test_auprc.reset()

    # ---------------------
    # Training Setup
    # ---------------------
    def configure_optimizers(self):
        """Called to configure the trainer's optimizer(s)."""
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer)
        metric_to_track = 'val_auroc'
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
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
    lit_dagnn = LitDAGNN(
        node_feat=karate_club_data_module.num_node_features,
        num_classes=karate_club_data_module.karate_club_dataset.num_classes,
        num_hidden_layers=args.num_hidden_layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        dropout_rate=args.dropout_rate)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.max_epochs = args.num_epochs

    # Initialize logger
    args.experiment_name = f'LitDAGNN-l{args.num_hidden_layers}-h{args.hidden_dim}-b{args.batch_size}' \
        if not args.experiment_name \
        else args.experiment_name

    # Log everything to Weights and Biases (WandB)
    run = wandb.init(name=args.experiment_name, project=args.project_name, entity=args.entity, reinit=True)
    logger = construct_wandb_pl_logger(args)

    # Assign specified logger (e.g. WandB) to Trainer instance
    trainer.logger = logger

    # ------------
    # Checkpoint
    # ------------
    # Resume from checkpoint if path to a valid one is provided
    args.ckpt_name = args.ckpt_name if args.ckpt_name is not None else 'LitDAGNN-{epoch:02d}-{val_auroc:.2f}.ckpt'
    checkpoint_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    trainer.resume_from_checkpoint = checkpoint_path if os.path.exists(checkpoint_path) else None

    # -----------
    # Training
    # -----------
    # Create and use callbacks
    early_stop_callback = EarlyStopping(monitor='val_auroc', mode='max', min_delta=0.00, patience=args.patience)
    checkpoint_callback = ModelCheckpoint(monitor='val_auroc', save_top_k=3, dirpath=args.ckpt_dir,
                                          filename='LitDAGNN-{epoch:02d}-{val_auroc:.2f}')
    lr_callback = LearningRateMonitor(logging_interval='epoch')  # Use with a learning rate scheduler
    trainer.callbacks = [early_stop_callback, checkpoint_callback, lr_callback]

    # Train with the provided model and data module
    trainer.fit(lit_dagnn, datamodule=karate_club_data_module)

    # -----------
    # Testing
    # -----------
    trainer.test()

    # ------------
    # Finalizing
    # ------------
    run.save(checkpoint_callback.best_model_path)
    run.finish()


if __name__ == '__main__':
    cli_main()
