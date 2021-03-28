import os

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.nn import Linear, ReLU, ModuleList
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from project.datasets.QM9.qm9_dgl_data_module import QM9DGLDataModule
from project.utils.fibers import Fiber
from project.utils.metrics import L1Loss, L2Loss
from project.utils.modules import GNormSE3, GConvSE3, GMaxPooling
from project.utils.utils import collect_args, process_args, get_basis_and_r, construct_wandb_pl_logger


class LitTFN(pl.LightningModule):
    """An SE(3)-equivariant GCN."""

    def __init__(self, num_layers: int, atom_feature_size: int, num_channels: int, num_nlayers: int = 1,
                 num_degrees: int = 4, edge_dim: int = 4, lr: float = 1e-3, num_epochs: int = 5, std: float = 1.0,
                 mean: float = 0.0, task: str = 'homo'):
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

        # Collect dataset-specific parameters
        self.std = std
        self.mean = mean
        self.task = task

        self.fibers = {'in': Fiber(1, atom_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, self.num_channels_out)}

        blocks = self._build_gcn(self.fibers, 1)
        self.block0, self.block1, self.block2 = blocks

        # Declare loss function(s) for training, validation, and testing
        self.L1Loss = L1Loss(self.std, self.mean, self.task)
        self.L2Loss = L2Loss()

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
        basis, r = get_basis_and_r(graph, self.num_degrees - 1, self.device)

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
        logits = self(graph)

        # Calculate the loss
        l1_loss, rescaled_l1_loss = self.L1Loss(logits, y)
        l2_loss = self.L2Loss(logits, y)

        # Log training metrics
        self.log('train_l1_loss', l1_loss, sync_dist=True)
        self.log('train_rescaled_l1_loss', rescaled_l1_loss, sync_dist=True)
        self.log('train_l2_loss', l2_loss, sync_dist=True)

        # Assemble and return the training step output
        output = {'loss': l1_loss}  # The loss key here is required
        return output

    def training_epoch_end(self, outs):
        """Lightning calls this at the end of every training epoch."""
        self.L1Loss.reset()
        self.L2Loss.reset()

    def validation_step(self, graph_and_y, batch_idx):
        """Lightning calls this inside the validation loop."""
        graph = graph_and_y[0]
        y = graph_and_y[1]

        # Make a forward pass through the network
        logits = self(graph)

        # Calculate the loss
        l1_loss, rescaled_l1_loss = self.L1Loss(logits, y)
        l2_loss = self.L2Loss(logits, y)

        # Log validation metrics
        self.log('val_l1_loss', l1_loss, sync_dist=True)
        self.log('val_rescaled_l1_loss', rescaled_l1_loss, sync_dist=True)
        self.log('val_l2_loss', l2_loss, sync_dist=True)

        # Assemble and return the validation step output
        output = {'loss': rescaled_l1_loss}  # The loss key here is required
        return output

    def validation_epoch_end(self, outs):
        """Lightning calls this at the end of every validation epoch."""
        self.L1Loss.reset()
        self.L2Loss.reset()

    def test_step(self, graph_and_y, batch_idx):
        """Lightning calls this inside the test loop."""
        graph = graph_and_y[0]
        y = graph_and_y[1]

        # Make a forward pass through the network
        logits = self(graph)

        # Calculate the loss
        l1_loss, rescaled_l1_loss = self.L1Loss(logits, y)
        l2_loss = self.L2Loss(logits, y)

        # Log test metrics
        self.log('test_l1_loss', l1_loss, sync_dist=True)
        self.log('test_rescaled_l1_loss', rescaled_l1_loss, sync_dist=True)
        self.log('test_l2_loss', l2_loss, sync_dist=True)

        # Assemble and return the test step output
        output = {'loss': rescaled_l1_loss}  # The loss key here is required
        return output

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
    data_module = QM9DGLDataModule(batch_size=args.batch_size, num_dataloader_workers=args.num_workers)
    data_module.prepare_data()
    data_module.setup()

    # -----------
    # Model
    # -----------
    lit_tfn = LitTFN(num_layers=args.num_layers,
                     atom_feature_size=data_module.num_node_features,
                     num_channels=args.num_channels,
                     num_nlayers=args.num_nlayers,
                     num_degrees=args.num_degrees,
                     edge_dim=data_module.num_edge_features,
                     lr=args.lr,
                     num_epochs=args.num_epochs,
                     std=data_module.std,
                     mean=data_module.mean,
                     task=args.task)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.max_epochs = args.num_epochs

    # Initialize logger
    args.experiment_name = f'TFN-d{args.num_degrees}-l{args.num_layers}-{args.num_channels}-{args.num_nlayers}' \
        if not args.experiment_name \
        else args.experiment_name

    # Log everything to Weights and Biases (WandB)
    run = wandb.init(name=args.experiment_name, project=args.project_name, entity=args.entity, reinit=True)
    logger = construct_wandb_pl_logger(args)

    # Assign specified logger (e.g. WandB) to Trainer instance
    trainer.logger = logger

    # -----------
    # Checkpoint
    # -----------
    # Resume from checkpoint if path to a valid one is provided
    args.ckpt_name = args.ckpt_name \
        if args.ckpt_name is not None \
        else 'LitTFN-{epoch:02d}-{val_rescaled_l1_loss:.2f}.ckpt'
    checkpoint_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    trainer.resume_from_checkpoint = checkpoint_path if os.path.exists(checkpoint_path) else None

    # -----------
    # Training
    # -----------
    # Create and use callbacks
    early_stop_callback = EarlyStopping(monitor='val_rescaled_l1_loss', mode='min', min_delta=0.00, patience=3)
    checkpoint_callback = ModelCheckpoint(monitor='val_rescaled_l1_loss', save_top_k=3, dirpath=args.ckpt_dir,
                                          filename='LitTFN-{epoch:02d}-{val_rescaled_l1_loss:.2f}')
    lr_callback = LearningRateMonitor(logging_interval='epoch')  # Use with a learning rate scheduler
    trainer.callbacks = [early_stop_callback, checkpoint_callback, lr_callback]

    # Train with the provided model and data module
    trainer.fit(lit_tfn, datamodule=data_module)

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
