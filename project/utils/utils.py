import os
from argparse import ArgumentParser

import dgl
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from scipy.constants import physical_constants

from project.utils.from_se3cnn.utils_steerable import _basis_transformation_Q_J, \
    precompute_sh, get_spherical_from_cartesian_torch
from project.utils.utils_profiling import profile

try:
    from types import SliceType
except ImportError:
    SliceType = slice

# __main__.pymol_argv = ['pymol', '-qc']  # Quiet and no GUI
# pymol.finish_launching()

hartree2eV = physical_constants['hartree-electron volt relationship'][0]
unit_conversion = {'mu': 1.0,
                   'alpha': 1.0,
                   'homo': hartree2eV,
                   'lumo': hartree2eV,
                   'gap': hartree2eV,
                   'r2': 1.0,
                   'zpve': hartree2eV,
                   'u0': hartree2eV,
                   'u298': hartree2eV,
                   'h298': hartree2eV,
                   'g298': hartree2eV,
                   'cv': 1.0}


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code derived from SE(3)-Transformer (https://github.com/FabianFuchsML/se3-transformer-public/):
# -------------------------------------------------------------------------------------------------------------------------------------

class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        return x @ Q


# Equivariant basis construction #
@profile
def get_basis(Y, max_degree):
    """Precompute the SE(3)-equivariant weight basis.
    This is called by get_basis_and_r().
    Args:
        Y: spherical harmonic dict, returned by utils_steerable.precompute_sh()
        max_degree: non-negative int for degree of highest feature type
    Returns:
        dict of equivariant bases, keys are in form '<d_in><d_out>'
    """
    device = Y[0].device
    # No need to backprop through the basis construction
    with torch.no_grad():
        basis = {}
        for d_in in range(max_degree + 1):
            for d_out in range(max_degree + 1):
                K_Js = []
                for J in range(abs(d_in - d_out), d_in + d_out + 1):
                    # Get spherical harmonic projection matrices
                    Q_J = _basis_transformation_Q_J(J, d_in, d_out)
                    Q_J = Q_J.float().T.to(device)

                    # Create kernel from spherical harmonics
                    K_J = torch.matmul(Y[J], Q_J)
                    K_Js.append(K_J)

                # Reshape so can take linear combinations with a dot product
                size = (-1, 1, 2 * d_out + 1, 1, 2 * d_in + 1, 2 * min(d_in, d_out) + 1)
                basis[f'{d_in},{d_out}'] = torch.stack(K_Js, -1).view(*size)
        return basis


def get_basis_and_r(G, max_degree):
    """Return equivariant weight basis (basis) and internodal distances (r).
    Call this function *once* at the start of each forward pass of the model.
    It computes the equivariant weight basis, W_J^lk(x), and internodal
    distances, needed to compute varphi_J^lk(x), of eqn 8 of
    https://arxiv.org/pdf/2006.10503.pdf. The return values of this function
    can be shared as input across all SE(3)-Transformer layers in a model.
    Args:
        G: DGL graph instance of type dgl.DGLGraph()
        max_degree: non-negative int for degree of highest feature-type
    Returns:
        dict of equivariant bases, keys are in form '<d_in><d_out>'
        vector of relative distances, ordered according to edge ordering of G
    """
    # Relative positional encodings (vector)
    r_ij = get_spherical_from_cartesian_torch(G.edata['d'])
    # Spherical harmonic basis
    Y = precompute_sh(r_ij, 2 * max_degree)
    # Equivariant basis (dict['d_in><d_out>'])
    basis = get_basis(Y, max_degree)
    # Relative distances (scalar)
    r = torch.sqrt(torch.sum(G.edata['d'] ** 2, -1, keepdim=True))
    return basis, r


def shape_is(a, b, ignore_batch=1):
    """
    check whether multi-dimensional array a has dimensions b; use in combination with assert
    :param a: multi dimensional array
    :param b: list of ints which indicate expected dimensions of a
    :param ignore_batch: if set to True, ignore first dimension of a
    :return: True or False
    """
    if ignore_batch:
        shape_a = np.array(a.shape[1:])
    else:
        shape_a = np.array(a.shape)
    shape_b = np.array(b)
    return np.array_equal(shape_a, shape_b)


def norm_with_epsilon(input_tensor, axis=None, keep_dims=False, epsilon=0.0):
    """
    Regularized norm
    Args:
        input_tensor: torch.Tensor
    Returns:
        torch.Tensor normed over axis
    """
    # return torch.sqrt(torch.max(torch.reduce_sum(torch.square(input_tensor), axis=axis, keep_dims=keep_dims), epsilon))
    keep_dims = bool(keep_dims)
    squares = torch.sum(input_tensor ** 2, axis=axis, keepdim=keep_dims)
    squares = torch.max(squares, torch.tensor([epsilon]).to(squares.device))
    return torch.sqrt(squares)


def collate(samples):
    graphs, y = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(y)


def norm2units(x, std, mean, task, denormalize=True, center=True):
    # Convert from normalized to QM9 representation
    if denormalize:
        x = x * std
        # Add the mean: not necessary for error computations
        if not center:
            x += mean
    x = unit_conversion[task] * x
    return x


def task_loss(pred, target, std, mean, task, use_mean=True):
    l1_loss = torch.sum(torch.abs(pred - target))
    l2_loss = torch.sum((pred - target) ** 2)
    if use_mean:
        l1_loss /= pred.shape[0]
        l2_loss /= pred.shape[0]

    rescale_loss = norm2units(l1_loss, std, mean, task)
    return l1_loss, l2_loss, rescale_loss


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for SGSET (https://github.com/jianlin-cheng/DeepInteract):
# -------------------------------------------------------------------------------------------------------------------------------------


def collect_args():
    """Collect all arguments required for training/testing."""
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    # -----------------
    # Model parameters
    # -----------------
    parser.add_argument('--num_layers', type=int, default=4, help="Number of equivariant layers")
    parser.add_argument('--num_degrees', type=int, default=4, help="Number of irreps {0,1,...,num_degrees-1}")
    parser.add_argument('--num_channels', type=int, default=16, help="Number of channels in middle layers")
    parser.add_argument('--num_nlayers', type=int, default=0, help="Number of layers for nonlinearity")
    parser.add_argument('--fully_connected', action='store_true', help="Include global node in graph")
    parser.add_argument('--div', type=float, default=4, help="Low dimensional embedding fraction")
    parser.add_argument('--pooling', type=str, default='avg', help="Choose from avg or max")
    parser.add_argument('--head', type=int, default=1, help="Number of attention heads")
    parser.add_argument('--geometric', type=bool, default=True, help="Whether to use a GVP for the FC layers")

    # -----------------
    # Meta-parameters
    # -----------------
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout (forget) rate")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of epochs")

    # -----------------
    # Data parameters
    # -----------------
    # parser.add_argument('--data_dir', type=str, default='datasets/DIPS/final/raw',
    #                     help='Path to final raw data directory for DIPS')
    # parser.add_argument('--data_dir', type=str, default='datasets/DB5/final/raw',
    #                     help='Path to final raw data directory for DB5')
    parser.add_argument('--data_dir', type=str, default='datasets/QM9/QM9_data.pt',
                        help='Path to preprocessed QM9 dataset')
    parser.add_argument('--task', type=str, default='homo',
                        help="QM9 task ['homo, 'mu', 'alpha', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']")

    # -----------------
    # Logging
    # -----------------
    parser.add_argument('--model', type=str, default='LitSGSET', help="Model being used")
    parser.add_argument('--name', type=str, default=None, help="Run name")
    parser.add_argument('--log_interval', type=int, default=25, help="Number of steps between logging key stats")
    parser.add_argument('--print_interval', type=int, default=250, help="Number of steps between printing key stats")
    parser.add_argument('--save_dir', type=str, default="models", help="Directory name to save models")
    parser.add_argument('--wandb', type=str, default='DeepInteract', help="WandB project name")

    # -----------------
    # Miscellaneous
    # -----------------
    parser.add_argument('--num_workers', type=int, default=24, help="Number of data loader workers")
    parser.add_argument('--profile', action='store_true', help="Exit after 10 steps for profiling")

    # -----------------
    # Seed parameter
    # -----------------
    parser.add_argument('--seed', type=int, default=None, help='Seed for NumPy and PyTorch')

    args, unparsed_argv = parser.parse_known_args()
    return args, unparsed_argv


def process_args(args, unparsed_argv):
    """Process all arguments required for training/testing."""
    # ---------------------------------------
    # Name fixing
    # ---------------------------------------
    if not args.name:
        args.name = f'SET-d{args.num_degrees}-l{args.num_layers}-{args.num_channels}-{args.num_nlayers}'
        # args.name = f'SGSET-d{args.num_degrees}-l{args.num_layers}-{args.num_channels}-{args.num_nlayers}'

    # ---------------------------------------
    # Model directory creation
    # ---------------------------------------
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # ---------------------------------------
    # Seed fixing for random numbers
    # ---------------------------------------
    if not args.seed:
        args.seed = 42  # np.random.randint(100000)
    pl.seed_everything(args.seed)

    # ---------------------------------------
    # Automatically choosing GPU if possible
    # ---------------------------------------
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    print("\n\nargs:", args)
    print("unparsed_argv:", unparsed_argv, "\n\n")


def construct_wandb_pl_logger(args):
    """Return an instance of WandbLogger with corresponding project and name strings."""
    return WandbLogger(name=args.name, project=args.wandb) if args.name else WandbLogger(project=f'{args.wandb}')
