import os
from argparse import ArgumentParser

import dgl
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from scipy.constants import physical_constants

from project.utils.from_se3cnn.utils_steerable import _basis_transformation_Q_J, \
    precompute_sh, get_spherical_from_cartesian_torch
from project.utils.utils_profiling import profile

try:
    from types import SliceType
except ImportError:
    SliceType = slice

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
def get_basis(Y, max_degree, device):
    """Precompute the SE(3)-equivariant weight basis.
    This is called by get_basis_and_r().
    Args:
        Y: spherical harmonic dict, returned by utils_steerable.precompute_sh()
        max_degree: non-negative int for degree of highest feature type
        device: Torch device for which basis is constructed
    Returns:
        dict of equivariant bases, keys are in form '<d_in><d_out>'
    """
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


def get_basis_and_r(G, max_degree, device):
    """Return equivariant weight basis (basis) and internodal distances (r).
    Call this function *once* at the start of each forward pass of the model.
    It computes the equivariant weight basis, W_J^lk(x), and internodal
    distances, needed to compute varphi_J^lk(x), of eqn 8 of
    https://arxiv.org/pdf/2006.10503.pdf. The return values of this function
    can be shared as input across all SE(3)-Transformer layers in a model.
    Args:
        G: DGL graph instance of type dgl.DGLGraph()
        max_degree: non-negative int for degree of highest feature-type
        device: Torch device for which basis is constructed
    Returns:
        dict of equivariant bases, keys are in form '<d_in><d_out>'
        vector of relative distances, ordered according to edge ordering of G
    """
    # Relative positional encodings (vector)
    r_ij = get_spherical_from_cartesian_torch(G.edata['d'])
    # Spherical harmonic basis
    Y = precompute_sh(r_ij, 2 * max_degree)
    # Equivariant basis (dict['d_in><d_out>'])
    basis = get_basis(Y, max_degree, device)
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


# TODO: Investigate collation of graphs for correct loss calculation
def collate(samples):
    graphs, y = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(y)


def rand_rot(x, dtype=np.float32):
    s = np.random.randn(3, 3)
    r, __ = np.linalg.qr(s)
    r = r.astype(dtype)
    return x @ r


def norm2units(x, std, mean, task, denormalize=True, center=True):
    # Convert from normalized to QM9 representation
    if denormalize:
        x = x * std
        # Add the mean: not necessary for error computations
        if not center:
            x += mean
    x = unit_conversion[task] * x
    return x


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code derived from egnn-pytorch (https://github.com/lucidrains/egnn-pytorch/blob/main/egnn_pytorch/utils.py):
# -------------------------------------------------------------------------------------------------------------------------------------

def fourier_encode_dist(x, num_encodings=4, include_self=True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x


def rot_z(gamma):
    return torch.tensor([
        [torch.cos(gamma), -torch.sin(gamma), 0],
        [torch.sin(gamma), torch.cos(gamma), 0],
        [0, 0, 1]
    ], dtype=gamma.dtype)


def rot_y(beta):
    return torch.tensor([
        [torch.cos(beta), 0, torch.sin(beta)],
        [0, 1, 0],
        [-torch.sin(beta), 0, torch.cos(beta)]
    ], dtype=beta.dtype)


def rot(alpha, beta, gamma):
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)


def batched_index_select(values, indices, dim=1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Equivariant-GNNs (https://github.com/amorehead/Equivariant-GNNs):
# -------------------------------------------------------------------------------------------------------------------------------------


def get_graph(src, dst, pos, node_feature, edge_feature, dtype, undirected=True, num_nodes=None):
    # src, dst : indices for vertices of source and destination, np.array
    # pos: x,y,z coordinates of all vertices with respect to the indices, np.array
    # node_feature: node feature of shape [num_atoms,node_feature_size,1], np.array
    # edge_feature: edge feature of shape [num_atoms,edge_feature_size], np.array
    if num_nodes:
        G = dgl.graph((src, dst), num_nodes=num_nodes)
    else:
        G = dgl.graph((src, dst))
    if undirected:
        G = dgl.to_bidirected(G)
    # Add node features to graph
    G.ndata['x'] = torch.tensor(pos.astype(dtype))  # [num_atoms,3]
    G.ndata['f'] = torch.tensor(node_feature.astype(dtype))
    # Add edge features to graph
    G.edata['w'] = torch.tensor(edge_feature.astype(dtype))  # [num_atoms,edge_feature_size]
    return G


def get_fully_connected_graph(pos, fill=0, dtype=np.float32):
    # pos :n by 3 np.array for xyz
    x = np.array(range(pos.shape[0]))
    src = np.repeat(x, x.shape[0])
    dst = np.tile(x, x.shape[0])
    flag = src != dst
    G = dgl.graph((src[flag], dst[flag]))
    G.ndata['x'] = pos
    G.ndata['f'] = torch.tensor(np.full((G.num_nodes(), 1, 1), fill).astype(dtype))
    G.edata['w'] = torch.tensor(np.full((G.num_edges(), 1), fill).astype(dtype))
    return G


def collect_args():
    """Collect all arguments required for training/testing."""
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    # -----------------
    # Model parameters
    # -----------------
    parser.add_argument('--num_layers', type=int, default=4, help="Number of equivariant layers")
    parser.add_argument('--num_degrees', type=int, default=4, help="Number of irreps {0,1,...,num_degrees-1}")
    parser.add_argument('--num_channels', type=int, default=32, help="Number of channels in middle layers")
    parser.add_argument('--num_nlayers', type=int, default=0, help="Number of layers for nonlinearity")
    parser.add_argument('--fully_connected', action='store_true', default=False, help="Include global node in graph")
    parser.add_argument('--div', type=float, default=2.0, help="Low dimensional embedding fraction")
    parser.add_argument('--pooling', type=str, default='max', help="Choose from avg or max")
    parser.add_argument('--head', type=int, default=1, help="Number of attention heads")
    parser.add_argument('--num_nearest_neighbors', type=int, default=3, help="Neighbor count threshold to define edges")

    # -----------------
    # Meta-parameters
    # -----------------
    parser.add_argument('--multi_gpu_backend', type=str, default='ddp', help="Backend to use for multi-GPU training")
    parser.add_argument('--num_gpus', type=int, default=-1, help="Number of GPUs to use (e.g. -1 = all available GPUs)")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout (forget) rate")
    parser.add_argument('--num_epochs', type=int, default=5, help="Number of epochs")
    parser.add_argument('--save_dir', type=str, default="models", help="Directory in which to save models")

    # -----------------
    # Data parameters
    # -----------------
    parser.add_argument('--data_dir', type=str, default='datasets/QM9/QM9_data.pt',
                        help='Path to preprocessed QM9 dataset')
    parser.add_argument('--task', type=str, default='homo',
                        help="QM9 task ['homo, 'mu', 'alpha', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']")

    # -----------------
    # Logging
    # -----------------
    parser.add_argument('--model', type=str, default='LitSET', help="Model being used")
    parser.add_argument('--log_interval', type=int, default=25, help="Number of steps between logging key stats")
    parser.add_argument('--print_interval', type=int, default=250, help="Number of steps between printing key stats")

    parser.add_argument('--experiment_name', type=str, default=None, help="Neptune experiment name")
    parser.add_argument('--project_name', type=str, default='amorehead/Equivariant-GNNs', help="Neptune project name")

    # -----------------
    # Miscellaneous
    # -----------------
    parser.add_argument('--num_workers', type=int, default=1, help="Number of data loader workers")
    parser.add_argument('--profile', action='store_true', default=False, help="Exit after 10 steps for profiling")

    # -----------------
    # Seed parameter
    # -----------------
    parser.add_argument('--seed', type=int, default=None, help='Seed for NumPy and PyTorch')

    # Parse all known arguments
    args, unparsed_argv = parser.parse_known_args()

    # Set HPC-specific parameter values
    args.accelerator = args.multi_gpu_backend
    args.gpus = args.num_gpus

    return args


def process_args(args):
    """Process all arguments required for training/testing."""
    # ---------------------------------------
    # Model directory creation
    # ---------------------------------------
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # ---------------------------------------
    # Seed fixing for random numbers
    # ---------------------------------------
    if not args.seed:
        args.seed = 42  # np.random.randint(100000)
    pl.seed_everything(args.seed)


def construct_neptune_pl_logger(args):
    """Return an instance of NeptuneLogger with corresponding project and experiment name strings."""
    return NeptuneLogger(experiment_name=args.experiment_name,
                         project_name=args.project_name,
                         close_after_fit=False,
                         params={'max_epochs': args.num_epochs, 'batch_size': args.batch_size, 'lr': args.lr},
                         tags=['pytorch-lightning', 'graph-neural-network', 'equivariance'],
                         upload_source_files=['*.py'])


def construct_tensorboard_pl_logger(args):
    """Return an instance of TensorBoardLogger with corresponding project and experiment name strings."""
    return TensorBoardLogger('tb_log', name=args.experiment_name)
