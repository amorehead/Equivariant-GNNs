import os

import pytorch_lightning as pl
from dgl.data import CoraGraphDataset
from dgl.nn.pytorch import GraphConv
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.nn.functional import relu, cross_entropy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from project.utils.utils import collect_args, process_args, construct_wandb_pl_logger


class LitEGNN(pl.LightningModule):
    """An E(n)-equivariant GNN."""

    def __init__(self, in_feats: int = 1, h_feats: int = 4, num_classes: int = 2,
                 num_layers: int = 4, num_channels: int = 16, pooling: str = 'avg',
                 lr: float = 1e-3, num_epochs: int = 5):
        """Initialize all the parameters for an EGNN."""
        super().__init__()
        self.save_hyperparameters()

        # Build the network
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.pooling = pooling
        self.lr = lr
        self.num_epochs = num_epochs

        # Assemble the layers of the network
        self.build_gnn_model()

        # Declare loss function(s) for training, validation, and testing
        self.cross_entropy = cross_entropy

    def build_gnn_model(self):
        """Define the layers of a single GNN."""
        # Marshal all equivariant layers
        self.conv1 = GraphConv(self.in_feats, self.h_feats)
        self.conv2 = GraphConv(self.h_feats, self.num_classes)

    # ---------------------
    # Training
    # ---------------------
    def forward(self, graph, in_feat):
        """Make a forward pass through the entire network."""
        h = self.conv1(graph, in_feat)
        h = relu(h)
        h = self.conv2(graph, h)
        return h

    def training_step(self, graph, batch_idx):
        """Lightning calls this inside the training loop."""
        features = graph.ndata['feat']
        labels = graph.ndata['label']
        train_mask = graph.ndata['train_mask']
        val_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']

        # Make a forward pass through the network
        logits = self.forward(graph, features)

        # Construct prediction
        pred = logits.argmax(1)

        # Calculate the loss - Note that you should only compute the losses of the nodes in the training set
        cross_entropy = self.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Log training metrics
        self.log('cross_entropy', cross_entropy)
        self.log('train_acc', train_acc)
        self.log('val_acc', val_acc)
        self.log('test_acc', test_acc)

        # Assemble and return the training step output
        output = {'loss': cross_entropy}  # The loss key here is required
        return output

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


def collate_fn(samples):
    graph = samples[0]
    return graph


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

    # -----------
    # Data
    # -----------
    # data_module = RGDGLDataModule(batch_size=args.batch_size,
    #                               num_dataloader_workers=args.num_workers,
    #                               seed=args.seed)
    # data_module.prepare_data()
    # data_module.setup()

    # Temporarily test with the Cora dataset
    dataset = CoraGraphDataset()
    train_dataloader = DataLoader(dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1,
                                  collate_fn=collate_fn)

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
        lit_egnn = LitEGNN(in_feats=dataset[0].ndata['feat'].shape[1],
                           h_feats=16,
                           num_classes=dataset.num_classes,
                           num_layers=args.num_layers,
                           num_channels=args.num_channels,
                           pooling=args.pooling,
                           lr=args.lr,
                           num_epochs=args.num_epochs)

    # -----------
    # Training
    # -----------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.min_epochs = args.num_epochs

    # Logging all args to wandb
    logger = construct_wandb_pl_logger(args)
    trainer.logger = logger

    trainer.fit(lit_egnn, train_dataloader=train_dataloader)

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
