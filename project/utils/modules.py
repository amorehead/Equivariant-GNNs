from typing import Dict

import dgl
import dgl.function as fn  # for graphs
import numpy as np
import torch
import torch.nn as nn
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling
from dgl.nn.pytorch.softmax import edge_softmax
from einops import rearrange
from packaging import version
from torch import Tensor, einsum, broadcast_tensors
from torch.nn import SiLU
from torch.nn.functional import normalize

from project.utils.fibers import Fiber, fiber2head
from project.utils.from_se3cnn.utils_steerable import _basis_transformation_Q_J, get_spherical_from_cartesian_torch, \
    precompute_sh
from project.utils.utils import fourier_encode_dist, batched_index_select
from project.utils.utils_profiling import profile  # load before other local modules


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code derived from SE(3)-Transformer (https://github.com/FabianFuchsML/se3-transformer-public/):
# -------------------------------------------------------------------------------------------------------------------------------------

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


### SE(3) equivariant operations on graphs in DGL

class GConvSE3(nn.Module):
    """A tensor field network layer as a DGL module.

    GConvSE3 stands for a Graph Convolution SE(3)-equivariant layer. It is the
    equivalent of a linear layer in an MLP, a conv layer in a CNN, or a graph
    conv layer in a GCN.
    At each node, the activations are split into different "feature types",
    indexed by the SE(3) representation type: non-negative integers 0, 1, 2, ..
    """

    def __init__(self, f_in, f_out, self_interaction: bool = False, edge_dim: int = 0):
        """SE(3)-equivariant Graph Conv Layer
        Args:
            f_in: list of tuples [(multiplicities, type),...]
            f_out: list of tuples [(multiplicities, type),...]
            self_interaction: include self-interaction in convolution
            edge_dim: number of dimensions for edge embedding
        """
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.edge_dim = edge_dim
        self.self_interaction = self_interaction

        # Neighbor -> center weights
        self.kernel_unary = nn.ModuleDict()
        for (mi, di) in self.f_in.structure:
            for (mo, do) in self.f_out.structure:
                self.kernel_unary[f'({di},{do})'] = PairwiseConv(di, mi, do, mo, edge_dim=edge_dim)

        # Center -> center weights
        self.kernel_self = nn.ParameterDict()
        if self_interaction:
            for m_in, d_in in self.f_in.structure:
                if d_in in self.f_out.degrees:
                    m_out = self.f_out.structure_dict[d_in]
                    W = nn.Parameter(torch.randn(1, m_out, m_in) / np.sqrt(m_in))
                    self.kernel_self[f'{d_in}'] = W

    def __repr__(self):
        return f'GConvSE3(structure={self.f_out}, self_interaction={self.self_interaction})'

    def udf_u_mul_e(self, d_out):
        """Compute the convolution for a single output feature type.
        This function is set up as a User Defined Function in DGL.
        Args:
            d_out: output feature type
        Returns:
            edge -> node function handle
        """

        def fnc(edges):
            # Neighbor -> center messages
            msg = 0
            for m_in, d_in in self.f_in.structure:
                src = edges.src[f'{d_in}'].view(-1, m_in * (2 * d_in + 1), 1)
                edge = edges.data[f'({d_in},{d_out})']
                msg = msg + torch.matmul(edge, src)
            msg = msg.view(msg.shape[0], -1, 2 * d_out + 1)

            # Center -> center messages
            if self.self_interaction:
                if f'{d_out}' in self.kernel_self.keys():
                    dst = edges.dst[f'{d_out}']
                    W = self.kernel_self[f'{d_out}']
                    msg = msg + torch.matmul(W, dst)

            return {'msg': msg.view(msg.shape[0], -1, 2 * d_out + 1)}

        return fnc

    @profile
    def forward(self, h, G=None, r=None, basis=None, **kwargs):
        """Forward pass of the linear layer
        Args:
            G: minibatch of (homo)graphs
            h: dict of features
            r: inter-atomic distances
            basis: pre-computed Q * Y
        Returns:
            tensor with new features [B, n_points, n_features_out]
        """
        with G.local_scope():
            # Add node features to local graph scope
            for k, v in h.items():
                G.ndata[k] = v

            # Add edge features
            if 'w' in G.edata.keys():
                w = G.edata['w']
                feat = torch.cat([w, r], -1)
            else:
                feat = torch.cat([r, ], -1)

            for (mi, di) in self.f_in.structure:
                for (mo, do) in self.f_out.structure:
                    etype = f'({di},{do})'
                    G.edata[etype] = self.kernel_unary[etype](feat, basis)

            # Perform message-passing for each output feature type
            for d in self.f_out.degrees:
                G.update_all(self.udf_u_mul_e(d), fn.mean('msg', f'out{d}'))

            return {f'{d}': G.ndata[f'out{d}'] for d in self.f_out.degrees}


class RadialFunc(nn.Module):
    """NN parameterized radial profile function."""

    def __init__(self, num_freq, in_dim, out_dim, edge_dim: int = 0):
        """NN parameterized radial profile function.
        Args:
            num_freq: number of output frequencies
            in_dim: multiplicity of input (num input channels)
            out_dim: multiplicity of output (num output channels)
            edge_dim: number of dimensions for edge embedding
        """
        super().__init__()
        self.num_freq = num_freq
        self.in_dim = in_dim
        self.mid_dim = 32
        self.out_dim = out_dim
        self.edge_dim = edge_dim

        self.net = nn.Sequential(nn.Linear(self.edge_dim + 1, self.mid_dim),
                                 BN(self.mid_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.mid_dim, self.mid_dim),
                                 BN(self.mid_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.mid_dim, self.num_freq * in_dim * out_dim))

        nn.init.kaiming_uniform_(self.net[0].weight)
        nn.init.kaiming_uniform_(self.net[3].weight)
        nn.init.kaiming_uniform_(self.net[6].weight)

    def __repr__(self):
        return f"RadialFunc(edge_dim={self.edge_dim}, in_dim={self.in_dim}, out_dim={self.out_dim})"

    def forward(self, x):
        y = self.net(x)
        return y.view(-1, self.out_dim, 1, self.in_dim, 1, self.num_freq)


class PairwiseConv(nn.Module):
    """SE(3)-equivariant convolution between two single-type features"""

    def __init__(self, degree_in: int, nc_in: int, degree_out: int,
                 nc_out: int, edge_dim: int = 0):
        """SE(3)-equivariant convolution between a pair of feature types.
        This layer performs a convolution from nc_in features of type degree_in
        to nc_out features of type degree_out.
        Args:
            degree_in: degree of input fiber
            nc_in: number of channels on input
            degree_out: degree of out order
            nc_out: number of channels on output
            edge_dim: number of dimensions for edge embedding
        """
        super().__init__()
        # Log settings
        self.degree_in = degree_in
        self.degree_out = degree_out
        self.nc_in = nc_in
        self.nc_out = nc_out

        # Functions of the degree
        self.num_freq = 2 * min(degree_in, degree_out) + 1
        self.d_out = 2 * degree_out + 1
        self.edge_dim = edge_dim

        # Radial profile function
        self.rp = RadialFunc(self.num_freq, nc_in, nc_out, self.edge_dim)

    @profile
    def forward(self, feat, basis):
        # Get radial weights
        R = self.rp(feat)
        kernel = torch.sum(R * basis[f'{self.degree_in},{self.degree_out}'], -1)
        return kernel.view(kernel.shape[0], self.d_out * self.nc_out, -1)


class G1x1SE3(nn.Module):
    """Graph Linear SE(3)-equivariant layer, equivalent to a 1x1 convolution.

    This is equivalent to a self-interaction layer in TensorField Networks.
    """

    def __init__(self, f_in, f_out, learnable=True):
        """SE(3)-equivariant 1x1 convolution.
        Args:
            f_in: input Fiber() of feature multiplicities and types
            f_out: output Fiber() of feature multiplicities and types
        """
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out

        # Linear mappings: 1 per output feature type
        self.transform = nn.ParameterDict()
        for m_out, d_out in self.f_out.structure:
            m_in = self.f_in.structure_dict[d_out]
            self.transform[str(d_out)] = nn.Parameter(torch.randn(m_out, m_in) / np.sqrt(m_in), requires_grad=learnable)

    def __repr__(self):
        return f"G1x1SE3(structure={self.f_out})"

    def forward(self, features, **kwargs):
        output = {}
        for k, v in features.items():
            if str(k) in self.transform.keys():
                output[k] = torch.matmul(self.transform[str(k)], v)
        return output


class GNormSE3(nn.Module):
    """Graph Norm-based SE(3)-equivariant nonlinearity.

    Nonlinearities are important in SE(3) equivariant GCNs. They are also quite
    expensive to compute, so it is convenient for them to share resources with
    other layers, such as normalization. The general workflow is as follows:
    > for feature type in features:
    >    norm, phase <- feature
    >    output = fnc(norm) * phase

    where fnc: {R+}^m -> R^m is a learnable map from m norms to m scalars.
    """

    def __init__(self, fiber, nonlin=nn.ReLU(inplace=True), num_layers: int = 0):
        """Initializer.
        Args:
            fiber: Fiber() of feature multiplicities and types
            nonlin: nonlinearity to use everywhere
            num_layers: non-negative number of linear layers in fnc
        """
        super().__init__()
        self.fiber = fiber
        self.nonlin = nonlin
        self.num_layers = num_layers

        # Regularization for computing phase: gradients explode otherwise
        self.eps = 1e-12

        # Norm mappings: 1 per feature type
        self.transform = nn.ModuleDict()
        for m, d in self.fiber.structure:
            self.transform[str(d)] = self._build_net(int(m))

    def __repr__(self):
        return f"GNormSE3(num_layers={self.num_layers}, nonlin={self.nonlin})"

    def _build_net(self, m: int):
        net = []
        for i in range(self.num_layers):
            net.append(BN(int(m)))
            net.append(self.nonlin)
            # TODO: implement cleaner init
            net.append(nn.Linear(m, m, bias=(i == self.num_layers - 1)))
            nn.init.kaiming_uniform_(net[-1].weight)
        if self.num_layers == 0:
            net.append(BN(int(m)))
            net.append(self.nonlin)
        return nn.Sequential(*net)

    @profile
    def forward(self, features, **kwargs):
        output = {}
        for k, v in features.items():
            # Compute the norms and normalized features
            # v shape: [...,m , 2*k+1]
            norm = v.norm(2, -1, keepdim=True).clamp_min(self.eps).expand_as(v)
            phase = v / norm

            # Transform on norms
            transformed = self.transform[str(k)](norm[..., 0]).unsqueeze(-1)

            # Nonlinearity on norm
            output[k] = (transformed * phase).view(*v.shape)

        return output


class BN(nn.Module):
    """SE(3)-equvariant batch/layer normalization"""

    def __init__(self, m):
        """SE(3)-equvariant batch/layer normalization
        Args:
            m: int for number of output channels
        """
        super().__init__()
        self.bn = nn.LayerNorm(m)

    def forward(self, x):
        return self.bn(x)


class GConvSE3Partial(nn.Module):
    """Graph SE(3)-equivariant node -> edge layer"""

    def __init__(self, f_in, f_out, edge_dim: int = 0):
        """SE(3)-equivariant partial convolution.
        A partial convolution computes the inner product between a kernel and
        each input channel, without summing over the result from each input
        channel. This unfolded structure makes it amenable to be used for
        computing the value-embeddings of the attention mechanism.
        Args:
            f_in: list of tuples [(multiplicities, type),...]
            f_out: list of tuples [(multiplicities, type),...]
        """
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.edge_dim = edge_dim

        # Node -> edge weights
        self.kernel_unary = nn.ModuleDict()
        for (mi, di) in self.f_in.structure:
            for (mo, do) in self.f_out.structure:
                self.kernel_unary[f'({di},{do})'] = PairwiseConv(di, mi, do, mo, edge_dim=edge_dim)

    def __repr__(self):
        return f'GConvSE3Partial(structure={self.f_out})'

    def udf_u_mul_e(self, d_out):
        """Compute the partial convolution for a single output feature type.
        This function is set up as a User Defined Function in DGL.
        Args:
            d_out: output feature type
        Returns:
            node -> edge function handle
        """

        def fnc(edges):
            # Neighbor -> center messages
            msg = 0
            for m_in, d_in in self.f_in.structure:
                src = edges.src[f'{d_in}'].view(-1, m_in * (2 * d_in + 1), 1)
                edge = edges.data[f'({d_in},{d_out})']
                msg = msg + torch.matmul(edge, src)
            msg = msg.view(msg.shape[0], -1, 2 * d_out + 1)

            return {f'out{d_out}': msg.view(msg.shape[0], -1, 2 * d_out + 1)}

        return fnc

    @profile
    def forward(self, h, G=None, r=None, basis=None, **kwargs):
        """Forward pass of the linear layer
        Args:
            h: dict of node-features
            G: minibatch of (homo)graphs
            r: inter-atomic distances
            basis: pre-computed Q * Y
        Returns:
            tensor with new features [B, n_points, n_features_out]
        """
        with G.local_scope():
            # Add node features to local graph scope
            for k, v in h.items():
                G.ndata[k] = v

            # Add edge features
            if 'w' in G.edata.keys():
                w = G.edata['w']  # shape: [#edges_in_batch, #bond_types]
                feat = torch.cat([w, r], -1)
            else:
                feat = torch.cat([r, ], -1)
            for (mi, di) in self.f_in.structure:
                for (mo, do) in self.f_out.structure:
                    etype = f'({di},{do})'
                    G.edata[etype] = self.kernel_unary[etype](feat, basis)

            # Perform message-passing for each output feature type
            for d in self.f_out.degrees:
                G.apply_edges(self.udf_u_mul_e(d))

            return {f'{d}': G.edata[f'out{d}'] for d in self.f_out.degrees}


class GMABSE3(nn.Module):
    """An SE(3)-equivariant multi-headed self-attention module for DGL graphs."""

    def __init__(self, f_value: Fiber, f_key: Fiber, n_heads: int):
        """SE(3)-equivariant MAB (multi-headed attention block) layer.
        Args:
            f_value: Fiber() object for value-embeddings
            f_key: Fiber() object for key-embeddings
            n_heads: number of heads
        """
        super().__init__()
        self.f_value = f_value
        self.f_key = f_key
        self.n_heads = n_heads
        self.new_dgl = version.parse(dgl.__version__) > version.parse('0.4.4')

    def __repr__(self):
        return f'GMABSE3(n_heads={self.n_heads}, structure={self.f_value})'

    def udf_u_mul_e(self, d_out):
        """Compute the weighted sum for a single output feature type.
        This function is set up as a User Defined Function in DGL.
        Args:
            d_out: output feature type
        Returns:
            edge -> node function handle
        """

        def fnc(edges):
            # Neighbor -> center messages
            attn = edges.data['a']
            value = edges.data[f'v{d_out}']

            # Apply attention weights
            msg = attn.unsqueeze(-1).unsqueeze(-1) * value

            return {'m': msg}

        return fnc

    @profile
    def forward(self, v, k: Dict = None, q: Dict = None, G=None, **kwargs):
        """Forward pass of the linear layer
        Args:
            G: minibatch of (homo)graphs
            v: dict of value edge-features
            k: dict of key edge-features
            q: dict of query node-features
        Returns:
            tensor with new features [B, n_points, n_features_out]
        """
        with G.local_scope():
            # Add node features to local graph scope
            ## We use the stacked tensor representation for attention
            for m, d in self.f_value.structure:
                G.edata[f'v{d}'] = v[f'{d}'].view(-1, self.n_heads, m // self.n_heads, 2 * d + 1)
            G.edata['k'] = fiber2head(k, self.n_heads, self.f_key, squeeze=True)
            G.ndata['q'] = fiber2head(q, self.n_heads, self.f_key, squeeze=True)

            # Compute attention weights
            ## Inner product between (key) neighborhood and (query) center
            G.apply_edges(fn.e_dot_v('k', 'q', 'e'))

            ## Apply softmax
            e = G.edata.pop('e')
            if self.new_dgl:
                # in dgl 5.3, e has an extra dimension compared to dgl 4.3
                # the following, we get rid of this be reshaping
                n_edges = G.edata['k'].shape[0]
                e = e.view([n_edges, self.n_heads])
            e = e / np.sqrt(self.f_key.n_features)
            G.edata['a'] = edge_softmax(G, e)

            # Perform attention-weighted message-passing
            for d in self.f_value.degrees:
                G.update_all(self.udf_u_mul_e(d), fn.sum('m', f'out{d}'))

            output = {}
            for m, d in self.f_value.structure:
                output[f'{d}'] = G.ndata[f'out{d}'].view(-1, m, 2 * d + 1)

            return output


class GSE3Res(nn.Module):
    """Graph attention block with SE(3)-equivariance and skip connection"""

    def __init__(self, f_in: Fiber, f_out: Fiber, edge_dim: int = 0, div: float = 4,
                 n_heads: int = 1, learnable_skip=True):
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.div = div
        self.n_heads = n_heads

        # f_mid_out has same structure as 'f_out' but #channels divided by 'div'
        # this will be used for the values
        f_mid_out = {k: int(v // div) for k, v in self.f_out.structure_dict.items()}
        self.f_mid_out = Fiber(dictionary=f_mid_out)

        # f_mid_in has same structure as f_mid_out, but only degrees which are in f_in
        # this will be used for keys and queries
        # (queries are merely projected, hence degrees have to match input)
        f_mid_in = {d: m for d, m in f_mid_out.items() if d in self.f_in.degrees}
        self.f_mid_in = Fiber(dictionary=f_mid_in)

        self.edge_dim = edge_dim

        self.GMAB = nn.ModuleDict()

        # Projections
        self.GMAB['v'] = GConvSE3Partial(f_in, self.f_mid_out, edge_dim=edge_dim)
        self.GMAB['k'] = GConvSE3Partial(f_in, self.f_mid_in, edge_dim=edge_dim)
        self.GMAB['q'] = G1x1SE3(f_in, self.f_mid_in)

        # Attention
        self.GMAB['attn'] = GMABSE3(self.f_mid_out, self.f_mid_in, n_heads=n_heads)

        # Skip connections
        self.project = G1x1SE3(self.f_mid_out, f_out, learnable=learnable_skip)
        self.add = GSum(f_out, f_in)
        # the following checks whether the skip connection would change
        # the output fibre structure; the reason can be that the input has
        # more channels than the output (for at least one degree); this would
        # then cause a (hard to debug) error in the next layer
        assert self.add.f_out.structure_dict == f_out.structure_dict, \
            'skip connection would change output structure'

    @profile
    def forward(self, features, G, **kwargs):
        # Embeddings
        v = self.GMAB['v'](features, G=G, **kwargs)
        k = self.GMAB['k'](features, G=G, **kwargs)
        q = self.GMAB['q'](features, G=G)

        # Attention
        z = self.GMAB['attn'](v, k=k, q=q, G=G)

        # Skip + residual
        z = self.project(z)
        z = self.add(z, features)
        return z


### Helper and wrapper functions

class GSum(nn.Module):
    """SE(3)-equivariant graph residual sum function."""

    def __init__(self, f_x: Fiber, f_y: Fiber):
        """SE(3)-equivariant graph residual sum function.
        Args:
            f_x: Fiber() object for fiber of summands
            f_y: Fiber() object for fiber of summands
        """
        super().__init__()
        self.f_x = f_x
        self.f_y = f_y
        self.f_out = Fiber.combine_max(f_x, f_y)

    def __repr__(self):
        return f"GSum(structure={self.f_out})"

    def forward(self, x, y):
        out = {}
        for k in self.f_out.degrees:
            k = str(k)
            if (k in x) and (k in y):
                if x[k].shape[1] > y[k].shape[1]:
                    diff = x[k].shape[1] - y[k].shape[1]
                    zeros = torch.zeros(x[k].shape[0], diff, x[k].shape[2]).to(y[k].device)
                    y[k] = torch.cat([y[k], zeros], 1)
                elif x[k].shape[1] < y[k].shape[1]:
                    diff = y[k].shape[1] - x[k].shape[1]
                    zeros = torch.zeros(x[k].shape[0], diff, x[k].shape[2]).to(y[k].device)
                    x[k] = torch.cat([x[k], zeros], 1)

                out[k] = x[k] + y[k]
            elif k in x:
                out[k] = x[k]
            elif k in y:
                out[k] = y[k]
        return out


class GAvgPooling(nn.Module):
    """Graph Average Pooling module."""

    def __init__(self, type='0'):
        super().__init__()
        self.pool = AvgPooling()
        self.type = type

    @profile
    def forward(self, features, G, **kwargs):
        if self.type == '0':
            h = features['0'][..., -1]
            pooled = self.pool(G, h)
        elif self.type == '1':
            pooled = []
            for i in range(3):
                h_i = features['1'][..., i]
                pooled.append(self.pool(G, h_i).unsqueeze(-1))
            pooled = torch.cat(pooled, axis=-1)
            pooled = {'1': pooled}
        else:
            print('GAvgPooling for type > 0 not implemented')
            exit()
        return pooled


class GMaxPooling(nn.Module):
    """Graph Max Pooling module."""

    def __init__(self):
        super().__init__()
        self.pool = MaxPooling()

    @profile
    def forward(self, features, G, **kwargs):
        h = features['0'][..., -1]
        return self.pool(G, h)


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code derived from egnn-pytorch (https://github.com/lucidrains/egnn-pytorch/blob/main/egnn_pytorch/egnn_pytorch.py):
# -------------------------------------------------------------------------------------------------------------------------------------

class GConvEn(nn.Module):
    """A graph neural network layer as a DGL module.

    GConvEn stands for a Graph Convolution E(n)-equivariant layer. It is the
    equivalent of a linear layer in an MLP, a conv layer in a CNN, or a graph
    conv layer in a GCN.
    """

    def __init__(
            self,
            node_feat,
            edge_feat=0,
            coord_feat=16,
            fourier_feat=0,
            norm_rel_coors=False,
            norm_coor_weights=False,
            num_nearest_neighbors=0,
            dropout=0.0
    ):
        """E(n)-equivariant Graph Conv Layer

        Parameters
        ----------
        node_feat : int
            Node feature size.
        edge_feat : int
            Edge feature size.
        coord_feat : int
            Coordinates feature size.
        fourier_feat : int
            Fourier feature size.
        """
        super().__init__()
        self.fourier_feat = fourier_feat

        edge_input_dim = (fourier_feat * 2) + (node_feat * 2) + edge_feat + 1
        dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # torch.Size([1, 32, 3, 33])
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * 2),
            dropout,
            SiLU(),
            nn.Linear(edge_input_dim * 2, coord_feat),
            SiLU()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(node_feat + coord_feat, node_feat * 2),
            dropout,
            SiLU(),
            nn.Linear(node_feat * 2, node_feat),
        )

        self.norm_coor_weights = norm_coor_weights
        self.norm_rel_coors = norm_rel_coors
        if norm_rel_coors:
            self.rel_coors_scale = nn.Parameter(torch.ones(1))

        last_coor_linear = nn.Linear(coord_feat * 4, 1)
        self.coors_mlp = nn.Sequential(
            nn.Linear(coord_feat, coord_feat * 4),
            dropout,
            SiLU(),
            last_coor_linear
        )

        # seems to be needed to keep the network from exploding to NaN with greater depths
        last_coor_linear.weight.data.fill_(0)

        self.num_nearest_neighbors = num_nearest_neighbors

    def forward(self, h, x, e=None, mask=None):
        """Forward pass of the linear layer

        Parameters
        ----------
        h : Tensor
            The input node embedding.
        x : Tensor
            The input coordinates embedding.
        e : Tensor
            The input edge embedding.
        mask : Tensor
            The coordinate mask to apply.
        """
        b, n, d, fourier_features, num_nearest = *h.shape, self.fourier_feat, self.num_nearest_neighbors
        use_nearest = num_nearest > 0

        rel_coors = rearrange(x, 'b i d -> b i () d') - rearrange(x, 'b j d -> b () j d')
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)

        i = j = n

        if use_nearest:
            nbhd_indices = rel_dist[..., 0].topk(num_nearest, dim=-1, largest=False).indices
            rel_coors = batched_index_select(rel_coors, nbhd_indices, dim=2)
            rel_dist = batched_index_select(rel_dist, nbhd_indices, dim=2)

            if e is not None:
                edges = batched_index_select(e, nbhd_indices, dim=2)

            j = num_nearest

        if fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings=fourier_features)
            rel_dist = rearrange(rel_dist, 'b i j () d -> b i j d')

        if use_nearest:
            feats_j = batched_index_select(h, nbhd_indices, dim=1)
        else:
            feats_j = rearrange(h, 'b j d -> b () j d')

        feats_i = rearrange(h, 'b i d -> b i () d')
        feats_i, feats_j = broadcast_tensors(feats_i, feats_j)

        edge_input = torch.cat((feats_i, feats_j, rel_dist), dim=-1)

        if e is not None:
            edge_input = torch.cat((edge_input, e), dim=-1)

        m_ij = self.edge_mlp(edge_input)

        coor_weights = self.coors_mlp(m_ij)
        coor_weights = rearrange(coor_weights, 'b i j () -> b i j')

        if self.norm_coor_weights:
            coor_weights = coor_weights.tanh()

        if self.norm_rel_coors:
            rel_coors = normalize(rel_coors, dim=-1) * self.rel_coors_scale

        if mask is not None:
            mask_i = rearrange(mask, 'b i -> b i ()')

            if use_nearest:
                mask_j = batched_index_select(mask, nbhd_indices, dim=1)
            else:
                mask_j = rearrange(mask, 'b j -> b () j')

            mask = mask_i * mask_j
            coor_weights.masked_fill_(~mask, 0.)

        coors_out = einsum('b i j, b i j c -> b i c', coor_weights, rel_coors) + x

        m_i = m_ij.sum(dim=-2)

        node_mlp_input = torch.cat((h, m_i), dim=-1)
        node_out = self.node_mlp(node_mlp_input) + h

        return node_out, coors_out

    def __repr__(self):
        return f'GConvEn(structure=h{self.node_feat}-x{self.coord_feat}-e{self.edge_feat})'


class GConvEnSparse(nn.Module):
    """A sparse graph neural network layer as a DGL module.

    GConvEnSparse stands for a sparse Graph Convolution E(n)-equivariant layer.
    It is the equivalent of a linear layer in an MLP, a conv layer in a CNN, or
    a graph conv layer in a GCN. This type of layer will scale more smoothly to
    larger quantities of graph nodes.
    """

    def __init__(self, node_feat: int, pos_feat: int = 3, edge_feat: int = 0,
                 coord_feat: int = 16, fourier_feat: int = 0, dropout=0):
        """Sparse E(n)-equivariant Graph Conv Layer

        Parameters
        ----------
        node_feat : int
            Node feature size.
        pos_feat : int
            Position feature size.
        edge_feat : int
            Edge feature size.
        coord_feat : int
            Coordinates feature size.
        fourier_feat : int
            Fourier feature size.
        dropout : float
            Dropout (forget) rate.
        """
        super().__init__()
        self.node_feat = node_feat
        self.pos_feat = pos_feat
        self.edge_feat = edge_feat
        self.coord_feat = coord_feat
        self.fourier_feat = fourier_feat

        self.edge_input_size = (self.fourier_feat * 2) + (self.node_feat * 2) + (self.edge_feat + 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_size, self.edge_input_size * 2),
            self.dropout,
            SiLU(),
            nn.Linear(self.edge_input_size * 2, self.coord_feat),
            SiLU()
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(self.coord_feat, self.coord_feat * 4),
            self.dropout,
            SiLU(),
            nn.Linear(self.coord_feat * 4, 1)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(self.node_feat + self.coord_feat, self.node_feat * 2),
            self.dropout,
            SiLU(),
            nn.Linear(self.node_feat * 2, self.node_feat),
        )

    @profile
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor = None, size: Tensor = None) -> Tensor:
        """Forward pass of the linear layer

        Parameters
        ----------
        x : Tensor
            A (num_points, d) tensor where d is pos_feat + node_feat.
        edge_index : Tensor
            The input edge indices tensor.
        edge_attr : Tensor
            A (num_edges, num_feats) tensor excluding basic distance feats.
        """
        x, coords = x[:, :-self.pos_feat], x[:, -self.pos_feat:]

        rel_coords = coords[edge_index[0]] - coords[edge_index[1]]
        rel_dist = (rel_coords ** 2).sum(dim=-1, keepdim=True) ** 0.5

        if self.fourier_feat > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings=self.fourier_feat)
            rel_dist = rearrange(rel_dist, 'n () d -> n d')

        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr = rel_dist

        hidden_out, coors_out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                                               coords=coords, rel_coords=rel_coords)
        return torch.cat([hidden_out, coors_out], dim=-1)

    def message(self, x_i, x_j, edge_attr) -> Tensor:
        m_ij = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        coord_w = self.coord_mlp(m_ij)
        return m_ij, coord_w

    def propagate(self, edge_index: Tensor, size: Tensor = None, **kwargs):
        """The initial call to start propagating messages

        Parameters
        ----------
        edge_index : Tensor
            A tensor that holds the indices of a general (sparse) assignment matrix of shape :obj:`[N, M]`.
        size : Tensor
            A (tuple, optional) tensor that, when None, will infer the size and be assumed to be quadratic.
        **kwargs : any
            Any additional data which is needed to construct and aggregate messages,
            and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__,
                                     edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        m_ij, coord_wij = self.message(**msg_kwargs)

        # Aggregate them
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        m_i = self.aggregate(m_ij, **aggr_kwargs)
        coord_wi = self.aggregate(coord_wij, **aggr_kwargs)
        coord_ri = self.aggregate(kwargs["rel_coords"], **aggr_kwargs)

        # Return tuple
        update_kwargs = self.inspector.distribute('update', coll_dict)

        node, coors = kwargs["x"], kwargs["coords"]
        coords_out = coors + (coord_wi * coord_ri)

        hidden_out = self.node_mlp(torch.cat([node, m_i], dim=-1))
        hidden_out = hidden_out + node

        return self.update((hidden_out, coords_out), **update_kwargs)

    def __repr__(self):
        return f'GConvEnSparse(structure=h{self.node_feat}-p{self.pos_feat}-e{self.edge_feat}-x{self.coord_feat})'


class GConvEnSparseNetwork(nn.Module):
    r"""Sample GNN model architecture that uses the GConvEnSparse
        message passing layer to learn over point clouds.
        Main MPNN layer introduced in https://arxiv.org/abs/2102.09844v1
        Inputs will be standard GNN: x, edge_index, edge_attr, batch, ...
        Args:
        * n_layers: int. number of MPNN layers
        * ... : same interpretation as the base layer.
        * embedding_nums: list. number of unique keys to embedd. for points
                          1 entry per embedding needed.
        * embedding_dims: list. point - number of dimensions of
                          the resulting embedding. 1 entry per embedding needed.
        * edge_embedding_nums: list. number of unique keys to embedd. for edges.
                               1 entry per embedding needed.
        * edge_embedding_dims: list. point - number of dimensions of
                               the resulting embedding. 1 entry per embedding needed.
        * recalc: bool. Whether to recalculate edge features between MPNN layers.
        * verbose: bool. verbosity level.
    """

    def __init__(self, n_layers, node_feat, pos_feat=3,
                 edge_feat=0, coord_feat=16,
                 fourier_feat=0,
                 embedding_nums=[], embedding_dims=[],
                 edge_embedding_nums=[], edge_embedding_dims=[],
                 recalc=True, verbose=False):
        super().__init__()

        self.n_layers = n_layers

        # Embeddings? Solve here.
        self.embedding_nums = embedding_nums
        self.embedding_dims = embedding_dims
        self.emb_layers = nn.ModuleList()
        self.edge_embedding_nums = edge_embedding_nums
        self.edge_embedding_dims = edge_embedding_dims
        self.edge_emb_layers = nn.ModuleList()

        # Instantiate point and edge embedding layers
        for i in range(len(self.embedding_dims)):
            self.emb_layers.append(nn.Embedding(num_embeddings=embedding_nums[i],
                                                embedding_dim=embedding_dims[i]))
            node_feat += embedding_dims[i] - 1

        for i in range(len(self.edge_embedding_dims)):
            self.edge_emb_layers.append(nn.Embedding(num_embeddings=edge_embedding_nums[i],
                                                     embedding_dim=edge_embedding_dims[i]))
            edge_feat += edge_embedding_dims[i] - 1
        # rest
        self.mpnn_layers = nn.ModuleList()
        self.node_feat = node_feat
        self.pos_feat = pos_feat
        self.edge_feat = edge_feat
        self.coord_feat = coord_feat
        self.fourier_feat = fourier_feat
        self.recalc = recalc
        self.verbose = verbose

        # instantiate layers
        for i in range(n_layers):
            layer = GConvEnSparse(node_feat=node_feat,
                                  pos_feat=pos_feat,
                                  edge_feat=edge_feat,
                                  coord_feat=coord_feat,
                                  fourier_feat=fourier_feat)
            self.mpnn_layers.append(layer)

    def forward(self, x, edge_index, batch, edge_attr,
                bsize=None, recalc_edge=None, verbose=0):
        """ Embedding of inputs when necessary, then pass layers.
            Recalculate edge features every time with the
            `recalc_edge` function if self.recalc_edge is set.
        """
        original_x = x.clone()
        original_edge_index = edge_index.clone()
        original_edge_attr = edge_attr.clone()
        # pick to embedd. embedd sequentially and add to input - points:
        to_embedd = x[:, -len(self.embedding_dims):].long()
        for i, emb_layer in enumerate(self.emb_layers):
            # the portion corresponding to `to_embedd` part gets dropped
            # at first iter
            stop_concat = -len(self.embedding_dims) if i == 0 else x.shape[-1]
            x = torch.cat([x[:, :stop_concat],
                           emb_layer(to_embedd[:, i])
                           ], dim=-1)
        # pass layers
        for i, layer in enumerate(self.mpnn_layers):
            # embedd edge items (needed everytime since edge_attr and idxs
            # are recalculated every pass)
            to_embedd = edge_attr[:, -len(self.edge_embedding_dims):].long()
            for i, edge_emb_layer in enumerate(self.edge_emb_layers):
                # the portion corresponding to `to_embedd` part gets dropped
                # at first iter
                stop_concat = -len(self.edge_embedding_dims) if i == 0 else x.shape[-1]
                edge_attr = torch.cat([edge_attr[:, :-len(self.edge_embedding_dims) + i],
                                       edge_emb_layer(to_embedd[:, i])
                                       ], dim=-1)
            # pass layers
            x = layer(x, edge_index, edge_attr, size=bsize)

            # recalculate edge info - not needed if last layer
            if i < len(self.mpnn_layers) - 1 and self.recalc:
                edge_attr, edge_index, _ = recalc_edge(x.detach())  # returns attr, idx, embedd_info
            else:
                edge_attr = original_edge_attr.clone()
                edge_index = original_edge_index.clone()

            if verbose:
                print("========")
                print(i, "layer, nlinks:", edge_attr.shape)

        return x

    def __repr__(self):
        return 'EGNN_Sparse_Network of: {0} layers'.format(len(self.mpnn_layers))

# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Equivariant-GNNs (https://github.com/amorehead/Equivariant-GNNs):
# -------------------------------------------------------------------------------------------------------------------------------------
