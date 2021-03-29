import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch import relu
from torch.nn import ModuleList, Module, CrossEntropyLoss
from torch.optim import Adam
from torchmetrics import Accuracy

from project.datasets.Cora.cora_dgl_data_module import CoraDGLDataModule
from project.utils.modules import MLPLayer, DAGNNConv
from project.utils.utils import collect_args, process_args, construct_wandb_pl_logger


class LitDAGNN(pl.LightningModule):
    """A Deep Adaptive Graph Neural Network (DAGNN) for PyTorch Lightning."""

    def __init__(self, node_feat: int = 29, num_classes: int = 2, num_hidden_layers: int = 12, hidden_dim: int = 32,
                 bias=True, activation: Module = relu, lr: float = 1e-3, num_epochs: int = 5,
                 dropout_rate: float = 0.5):
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
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate

        # Assemble the layers of the network
        self.build_gnn_model()

        # Declare loss function(s) for training, validation, and testing
        self.ce = CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def build_gnn_model(self):
        """Define the MLP layers of a LitDAGNN."""
        # Marshal all MLP layers
        self.mlp_block = ModuleList([MLPLayer(in_dim=self.node_feat, out_dim=self.hidden_dim, bias=self.bias,
                                              activation=self.activation, dropout=self.dropout_rate),
                                     MLPLayer(in_dim=self.hidden_dim, out_dim=self.node_feat, bias=self.bias,
                                              activation=None, dropout=self.dropout_rate)])
        self.dagnn = DAGNNConv(in_dim=self.node_feat, k=self.num_hidden_layers)

    # ---------------------
    # Training
    # ---------------------
    def gnn_forward(self, graph, feats):
        """Make a forward pass through the entire network."""
        for layer in self.mlp_block:
            feats = layer(feats)
        feats = self.dagnn(graph, feats)
        return feats

    def forward(self, graph, feats):
        """Make a forward pass through the entire network."""
        # Forward propagate with both GNNs - must add self loops to avoid exploding logit values
        logits = self.gnn_forward(graph.add_self_loop(), feats)

        # Return network prediction
        return logits

    def training_step(self, batch, batch_idx):
        """Lightning calls this inside the training loop."""
        graph = batch
        labels = graph.ndata['label']
        train_mask = graph.ndata['train_mask']
        val_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']

        # Make a forward pass through the network for an entire batch of training graph pairs
        logits = self(graph, graph.ndata['feat'])

        # Compute prediction
        preds = logits.argmax(1)

        # Calculate the batch loss
        loss = self.ce(logits[train_mask], labels[train_mask])  # Calculate CrossEntropyLoss of a single batch

        # Log training step metric(s)
        self.log('train_ce', loss, on_step=True, sync_dist=True)
        self.log('train_acc', self.train_acc(preds[train_mask], labels[train_mask]), on_step=True, sync_dist=True)
        self.log('val_acc', self.val_acc(preds[val_mask], labels[val_mask]), on_step=True, sync_dist=True)
        self.log('test_acc', self.test_acc(preds[test_mask], labels[test_mask]), on_step=True, sync_dist=True)

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        self.log('train_acc', self.train_acc.compute(), sync_dist=True)
        self.log('val_acc', self.val_acc.compute(), sync_dist=True)
        self.log('test_acc', self.test_acc.compute(), sync_dist=True)
        self.train_acc.reset()
        self.val_acc.reset()
        self.test_acc.reset()

    # ---------------------
    # Training Setup
    # ---------------------
    def configure_optimizers(self):
        """Called to configure the trainer's optimizer(s)."""
        optimizer = Adam(self.parameters(), lr=self.lr)
        metric_to_track = 'val_acc'
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
    cora_data_module = CoraDGLDataModule(batch_size=1, num_dataloader_workers=args.num_workers)
    cora_data_module.prepare_data()
    cora_data_module.setup()

    # ------------
    # Model
    # ------------
    lit_dagnn = LitDAGNN(
        node_feat=cora_data_module.num_node_features,
        num_classes=cora_data_module.cora_graph_dataset.num_classes,
        num_hidden_layers=args.num_layers,
        lr=args.lr,
        num_epochs=args.num_epochs,
        dropout_rate=args.dropout_rate)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.max_epochs = args.num_epochs

    # Initialize logger
    args.experiment_name = f'LitDAGNN-l{args.num_layers}-h{args.num_channels}-b{args.batch_size}' \
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
    args.ckpt_name = args.ckpt_name if args.ckpt_name is not None else 'LitDAGNN-{epoch:02d}-{val_acc:.2f}.ckpt'
    checkpoint_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    trainer.resume_from_checkpoint = checkpoint_path if os.path.exists(checkpoint_path) else None

    # -----------
    # Training
    # -----------
    # Create and use callbacks
    early_stop_callback = EarlyStopping(monitor='val_acc', mode='max', min_delta=0.00, patience=args.patience)
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_top_k=3, dirpath=args.ckpt_dir,
                                          filename='LitDAGNN-{epoch:02d}-{val_acc:.2f}')
    lr_callback = LearningRateMonitor(logging_interval='epoch')  # Use with a learning rate scheduler
    trainer.callbacks = [early_stop_callback, checkpoint_callback, lr_callback]

    # Train with the provided model and data module
    trainer.fit(lit_dagnn, datamodule=cora_data_module)

    # -----------
    # Testing
    # -----------
    trainer.test()


if __name__ == '__main__':
    cli_main()
