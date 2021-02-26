import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.metrics import MeanSquaredError
from torch.nn import ModuleList
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from project.datasets.RG.rg_dgl_data_module import RGDGLDataModule
from project.utils.utils import collect_args, process_args, construct_wandb_pl_logger


class LitEGNN(pl.LightningModule):
    """An E(n)-equivariant GNN."""

    def __init__(self, num_layers: int, num_channels: int, pooling: str = 'avg', lr: float = 1e-3, num_epochs: int = 5):
        """Initialize all the parameters for an EGNN."""
        super().__init__()
        self.save_hyperparameters()

        # Build the network
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.pooling = pooling
        self.lr = lr
        self.num_epochs = num_epochs

        # Assemble the layers of the network
        self.g_block, self.fc_block = self.build_gnn_model()

        # Declare loss function(s) for training, validation, and testing
        self.MSE = MeanSquaredError()

    def build_gnn_model(self):
        """Define the layers of a single GNN."""
        # Marshal all equivariant layers
        g_block = []
        fc_block = []
        return ModuleList(g_block), ModuleList(fc_block)

    # ---------------------
    # Training
    # ---------------------
    def forward(self, graph):
        """Make a forward pass through the entire network."""
        # Encoder (equivariant layers)
        h = {'0': graph.ndata['f']}
        for layer in self.g_block:
            h = layer(h, G=graph)
        for layer in self.fc_block:
            h = layer(h)
        return h

    def training_step(self, graph_and_y, batch_idx):
        """Lightning calls this inside the training loop."""
        graph = graph_and_y[0]
        y = graph_and_y[1]

        # Make a forward pass through the network
        logits = self.forward(graph)

        # Calculate the loss
        mse = self.MSE(logits, y)

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
        mse = self.MSE(logits, y)

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
        mse = self.MSE(logits, y)

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

    def configure_callbacks(self):
        early_stop = EarlyStopping(monitor="val_mse", mode="max")
        checkpoint = ModelCheckpoint(monitor="val_mse", save_top_k=1)
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

    # -----------
    # Data
    # -----------
    data_module = RGDGLDataModule(batch_size=args.batch_size,
                                  num_dataloader_workers=args.num_workers,
                                  seed=args.seed)
    data_module.prepare_data()
    data_module.setup()

    # -----------
    # Model
    # -----------
    lit_egnn = LitEGNN(num_layers=args.num_layers,
                       num_channels=args.num_channels,
                       pooling=args.pooling,
                       lr=args.lr,
                       num_epochs=args.num_epochs)

    # ------------
    # Checkpoint
    # ------------
    checkpoint_save_path = os.path.join(args.save_dir, f'{args.name}.pth')
    try:
        lit_egnn = LitEGNN.load_from_checkpoint(checkpoint_save_path)
        print(f'Resuming from checkpoint {checkpoint_save_path}\n')
    except:
        print(f'Could not restore checkpoint {checkpoint_save_path}. Skipping...\n')

    # -----------
    # Training
    # -----------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.max_epochs = args.num_epochs

    # Logging all args to wandb
    logger = construct_wandb_pl_logger(args)
    trainer.logger = logger

    trainer.fit(lit_egnn, datamodule=data_module)

    # -----------
    # Testing
    # -----------
    rg_test_results = trainer.test()
    print(f'Model testing results on dataset: {rg_test_results}\n')

    # ------------
    # Finalizing
    # ------------
    print(f'Saving checkpoint {checkpoint_save_path}\n')
    trainer.save_checkpoint(checkpoint_save_path)


if __name__ == '__main__':
    cli_main()
