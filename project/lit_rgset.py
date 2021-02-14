import os

import pytorch_lightning as pl
from project.datasets.QM9.qm9_dgl_data_module import QM9DGLDataModule
from project.utils.fibers import Fiber
from project.utils.modules import GAvgPooling, GSE3Res, GNormSE3, GConvSE3, GMaxPooling
from project.utils.utils import collect_args, process_args, get_basis_and_r, task_loss, construct_wandb_pl_logger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import Linear, ReLU, ModuleList
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class LitRGSET(pl.LightningModule):
    """An SE(3)-equivariant GCN with attention for random graphs."""

    def __init__(self, num_layers: int, atom_feature_size: int, num_channels: int, num_nlayers: int = 1,
                 num_degrees: int = 4, edge_dim: int = 4, div: float = 4, pooling: str = 'avg', n_heads: int = 1,
                 geometric: bool = True, lr: float = 1e-3, num_epochs: int = 5, std: float = 1.0, mean: float = 0.0,
                 task: str = 'homo'):
        """Initialize all the parameters for a RGSET."""
        super().__init__()
        self.save_hyperparameters()

        # Build the siamese networks
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.pooling = pooling
        self.n_heads = n_heads
        self.geometric = geometric
        self.lr = lr
        self.num_epochs = num_epochs

        # Capture dataset specific parameters for loss measurement
        self.std = std
        self.mean = mean
        self.task = task

        # Assemble the layers of the network
        self.fibers = {'in': Fiber(1, atom_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, num_degrees * self.num_channels)}
        blocks = self.build_gcn_model(self.fibers, 1)
        self.Gblock, self.FCblock = blocks

        # Declare loss function for training, validation, and testing
        self.loss = task_loss

    def build_gcn_model(self, fibers, out_dim):
        """Define the layers of a single GCN."""
        # Marshal all equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim, div=self.div, n_heads=self.n_heads))
            Gblock.append(GNormSE3(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        # Pooling
        if self.pooling == 'avg':
            Gblock.append(GAvgPooling())
        elif self.pooling == 'max':
            Gblock.append(GMaxPooling())

        # FC layers
        FCblock = [Linear(self.fibers['out'].n_features, self.fibers['out'].n_features),
                   ReLU(inplace=True),
                   Linear(self.fibers['out'].n_features, out_dim)]

        return ModuleList(Gblock), ModuleList(FCblock)

    # ---------------------
    # Training
    # ---------------------
    def gcn_forward(self, G):
        """Make a forward pass through a single GCN."""
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees - 1)

        # Encoder (equivariant layers)
        h = {'0': G.ndata['f']}
        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)

        for layer in self.FCblock:
            h = layer(h)

        return h

    def forward(self, graph):
        """Make a forward pass through the entire network."""
        # Forward propagate with GCN
        logits = self.gcn_forward(graph)

        # Return network prediction
        return logits

    def training_step(self, graph_and_y, batch_idx):
        """Lightning calls this inside the training loop."""
        graph = graph_and_y[0]
        y = graph_and_y[1]

        # Make a forward pass through the network
        logits = self.forward(graph)

        # Calculate the loss
        l1_loss, _, rescale_loss = self.loss(logits, y, self.std, self.mean, self.task)

        # Log training metrics
        self.log('train_l1_loss', l1_loss)
        self.log('train_rescale_loss', rescale_loss)

        # Assemble and return the training step output
        output = {'loss': l1_loss}  # The loss key here is required
        return output

    def validation_step(self, graph_and_y, batch_idx):
        """Lightning calls this inside the validation loop."""
        graph = graph_and_y[0]
        y = graph_and_y[1]

        # Make a forward pass through the network
        logits = self.forward(graph)

        # Calculate the loss
        _, _, rescale_loss = self.loss(logits, y, self.std, self.mean, self.task)

        # Log training metrics
        self.log('val_rescale_loss', rescale_loss)

        # Assemble and return the training step output
        output = {'loss': rescale_loss}  # The loss key here is required
        return output

    def test_step(self, graph_and_y, batch_idx):
        """Lightning calls this inside the test loop."""
        graph = graph_and_y[0]
        y = graph_and_y[1]

        # Make a forward pass through the network
        logits = self.forward(graph)

        # Calculate the loss
        _, _, rescale_loss = self.loss(logits, y, self.std, self.mean, self.task)

        # Log training metrics
        self.log('test_rescale_loss', rescale_loss)

        # Assemble and return the training step output
        output = {'loss': rescale_loss}  # The loss key here is required
        return output

    # ---------------------
    # Training Setup
    # ---------------------
    def configure_optimizers(self):
        """Called to configure the trainer's optimizer(s)."""
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer, self.num_epochs, eta_min=1e-4)
        metric_to_track = 'val_loss'
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': metric_to_track
        }


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
    qm9_data_module = QM9DGLDataModule(data_dir=args.data_dir,
                                       task=args.task,
                                       batch_size=args.batch_size,
                                       num_dataloader_workers=args.num_workers,
                                       seed=args.seed)
    qm9_data_module.setup()
    qm9_data_module.prepare_data()

    # -----------
    # Model
    # -----------
    lit_rgset = LitRGSET(num_layers=args.num_layers,
                         atom_feature_size=qm9_data_module.num_atom_features,
                         num_channels=args.num_channels,
                         num_nlayers=args.num_nlayers,
                         num_degrees=args.num_degrees,
                         edge_dim=qm9_data_module.num_bonds,
                         div=args.div,
                         pooling=args.pooling,
                         n_heads=args.head,
                         geometric=args.geometric,
                         lr=args.lr,
                         std=qm9_data_module.qm9_train.std,
                         mean=qm9_data_module.qm9_train.mean,
                         task=args.task)

    # ------------
    # Checkpoint
    # ------------
    checkpoint_save_path = os.path.join(args.save_dir, f'{args.name}.pth')
    try:
        lit_rgset = LitRGSET.load_from_checkpoint(f'{args.name}.pth')
        print(f'Resuming from checkpoint {checkpoint_save_path}\n')
    except:
        print(f'Could not restore checkpoint {checkpoint_save_path}. Skipping...\n')

    # -----------
    # Training
    # -----------
    trainer = pl.Trainer.from_argparse_args(args)
    checkpoint_callback = ModelCheckpoint(monitor='val_auroc_score', save_top_k=1)
    trainer.callbacks = [checkpoint_callback]

    # Logging all args to wandb
    logger = construct_wandb_pl_logger(args)
    trainer.logger = logger

    trainer.fit(lit_rgset, datamodule=qm9_data_module)

    # -----------
    # Testing
    # -----------
    rg_test_results = trainer.test()
    print(f'Model testing results on RG dataset: {rg_test_results}\n')

    # ------------
    # Finalizing
    # ------------
    print(f'Saving checkpoint {checkpoint_save_path}\n')
    trainer.save_checkpoint(checkpoint_save_path)


if __name__ == '__main__':
    cli_main()