import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics import MeanSquaredError
from torch.nn import Linear, ReLU, ModuleList
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from project.datasets.RG.rg_dgl_data_module import RGDGLDataModule
from project.utils.fibers import Fiber
from project.utils.modules import GNormSE3, GConvSE3, GMaxPooling
from project.utils.utils import collect_args, process_args, get_basis_and_r


class LitRGTFN(pl.LightningModule):
    """An SE(3)-equivariant GCN for random graphs."""

    def __init__(self, num_layers: int, atom_feature_size: int,
                 num_channels: int, num_nlayers: int = 1, num_degrees: int = 4,
                 edge_dim: int = 4, lr: float = 1e-3, num_epochs: int = 5, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.num_channels_out = num_channels * num_degrees
        self.edge_dim = edge_dim
        self.lr = lr
        self.num_epochs = num_epochs

        self.fibers = {'in': Fiber(1, atom_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, self.num_channels_out)}

        blocks = self._build_gcn(self.fibers, 1)
        self.block0, self.block1, self.block2 = blocks

        # Declare loss function(s) for training, validation, and testing
        self.loss = MeanSquaredError()

    def _build_gcn(self, fibers, out_dim):
        block0 = []
        fin = fibers['in']
        for i in range(self.num_layers - 1):
            block0.append(GConvSE3(fin, fibers['mid'], self_interaction=True, edge_dim=self.edge_dim))
            block0.append(GNormSE3(fibers['mid'], num_layers=self.num_nlayers))
            fin = fibers['mid']
        block0.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        block1 = [GMaxPooling()]

        block2 = []
        block2.append(Linear(self.num_channels_out, self.num_channels_out))
        block2.append(ReLU(inplace=True))
        block2.append(Linear(self.num_channels_out, out_dim))

        return ModuleList(block0), ModuleList(block1), ModuleList(block2)

    # ---------------------
    # Training
    # ---------------------
    def forward(self, graph):
        """Make a forward pass through the entire network."""
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(graph, self.num_degrees - 1)

        # encoder (equivariant layers)
        h = {'0': graph.ndata['f']}
        for layer in self.block0:
            h = layer(h, G=graph, r=r, basis=basis)

        for layer in self.block1:
            h = layer(h, graph)

        for layer in self.block2:
            h = layer(h)

        return h

    def training_step(self, graph_and_y, batch_idx):
        """Lightning calls this inside the training loop."""
        graph = graph_and_y[0]
        y = graph_and_y[1]

        # Make a forward pass through the network
        logits = self.forward(graph)

        # Calculate the loss
        mse = self.loss(logits, y)

        # Log training metrics
        self.log('train_mse', mse)

        # Assemble and return the training step output
        output = {'loss': mse}  # The loss key here is required
        return output

    def validation_step(self, graph_and_y, batch_idx):
        """Lightning calls this inside the validation loop."""
        graph = graph_and_y[0]
        y = graph_and_y[1]

        # Make a forward pass through the network
        logits = self.forward(graph)

        # Calculate the loss
        mse = self.loss(logits, y)

        # Log validation metrics
        self.log('val_mse', mse)

        # Assemble and return the validation step output
        output = {'loss': mse}  # The loss key here is required
        return output

    def test_step(self, graph_and_y, batch_idx):
        """Lightning calls this inside the test loop."""
        graph = graph_and_y[0]
        y = graph_and_y[1]

        # Make a forward pass through the network
        logits = self.forward(graph)

        # Calculate the loss
        mse = self.loss(logits, y)

        # Log test metrics
        self.log('test_mse', mse)

        # Assemble and return the test step output
        output = {'loss': mse}  # The loss key here is required
        return output

    # ---------------------
    # Training Setup
    # ---------------------
    def configure_optimizers(self):
        """Called to configure the trainer's optimizer(s)."""
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer, self.num_epochs, eta_min=1e-4)
        metric_to_track = 'val_mse'
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
    rg_data_module = RGDGLDataModule(node_feature_size=args.node_feature_size,
                                     edge_feature_size=args.edge_feature_size,
                                     batch_size=args.batch_size,
                                     num_dataloader_workers=args.num_workers,
                                     seed=args.seed)
    rg_data_module.setup()
    rg_data_module.prepare_data()

    # -----------
    # Model
    # -----------
    lit_rgtfn = LitRGTFN(num_layers=args.num_layers,
                         atom_feature_size=rg_data_module.num_node_features,
                         num_channels=args.num_channels,
                         num_nlayers=args.num_nlayers,
                         num_degrees=args.num_degrees,
                         edge_dim=rg_data_module.num_edge_features,
                         lr=args.lr,
                         num_epochs=args.num_epochs)

    # ------------
    # Checkpoint
    # ------------
    checkpoint_save_path = os.path.join(args.save_dir, f'{args.name}.pth')
    try:
        lit_rgtfn = LitRGTFN.load_from_checkpoint(f'{args.name}.pth')
        print(f'Resuming from checkpoint {checkpoint_save_path}\n')
    except:
        print(f'Could not restore checkpoint {checkpoint_save_path}. Skipping...\n')

    # -----------
    # Training
    # -----------
    trainer = pl.Trainer.from_argparse_args(args)
    checkpoint_callback = ModelCheckpoint(monitor='val_mse', save_top_k=1)
    trainer.callbacks = [checkpoint_callback]

    # Logging all args to wandb
    # logger = construct_wandb_pl_logger(args)
    # trainer.logger = logger

    trainer.fit(lit_rgtfn, datamodule=rg_data_module)

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
