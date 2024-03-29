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
from torch import Tensor, einsum, broadcast_tensors, relu, sigmoid
from torch.nn import GELU
from torch.nn.functional import normalize
from torch.nn.parameter import Parameter

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

class EnInvGraphConv(nn.Module):
    """A graph neural network layer as a DGL module.

    EnInvGraphConv stands for a Graph Convolution E(n)-invariant layer. It is the
    equivalent of a linear layer in an MLP, a conv layer in a CNN, or a graph
    conv layer in a GCN.
    """

    def __init__(
            self,
            node_feat,
            edge_feat=0,
            coord_feat=16,
            fourier_feat=0,
            norm_rel_coords=False,
            norm_coord_weights=False,
            num_nearest_neighbors=0,
            dropout=0.0,
            init_eps=1e-3
    ):
        """E(n)-invariant Graph Conv Layer

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
        norm_rel_coords : boolean
            Fourier feature size.
        norm_coord_weights : boolean
            Fourier feature size.
        num_nearest_neighbors : int
            Fourier feature size.
        dropout : float
            Fourier feature size.
        init_eps : float
            Fourier feature size.
        """
        super().__init__()
        self.fourier_feat = fourier_feat

        edge_input_dim = (fourier_feat * 2) + (node_feat * 2) + edge_feat + 1
        dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * 2),
            dropout,
            GELU(),
            nn.Linear(edge_input_dim * 2, coord_feat),
            GELU()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(node_feat + coord_feat, node_feat * 2),
            dropout,
            GELU(),
            nn.Linear(node_feat * 2, node_feat),
        )

        self.norm_coord_weights = norm_coord_weights
        self.norm_rel_coords = norm_rel_coords

        if norm_rel_coords:
            self.rel_coords_scale = nn.Parameter(torch.ones(1))

        self.coords_mlp = nn.Sequential(
            nn.Linear(coord_feat, coord_feat * 4),
            dropout,
            GELU(),
            nn.Linear(coord_feat * 4, 1)
        )

        self.num_nearest_neighbors = num_nearest_neighbors

        self.init_eps = init_eps
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # Seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.normal_(module.weight, std=self.init_eps)

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
        nbhd_indices = None

        rel_coords = rearrange(x, 'b i d -> b i () d') - rearrange(x, 'b j d -> b () j d')
        rel_dist = (rel_coords ** 2).sum(dim=-1, keepdim=True)

        if use_nearest:
            nbhd_indices = rel_dist[..., 0].topk(num_nearest, dim=-1, largest=False).indices
            rel_coords = batched_index_select(rel_coords, nbhd_indices, dim=2)
            rel_dist = batched_index_select(rel_dist, nbhd_indices, dim=2)

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

        m_i = m_ij.sum(dim=-2)

        node_mlp_input = torch.cat((h, m_i), dim=-1)
        node_out = self.node_mlp(node_mlp_input) + h

        # Free GPU memory
        rel_coords.detach()
        rel_dist.detach()
        feats_i.detach()
        feats_j.detach()
        edge_input.detach()
        m_i.detach()
        m_ij.detach()
        node_mlp_input.detach()
        if nbhd_indices is not None:
            nbhd_indices.detach()
        if mask is not None:
            mask.detach()

        return node_out

    def __repr__(self):
        return f'EnInvGraphConv(structure=h{self.node_feat}-x{self.coord_feat}-e{self.edge_feat})'


class EnGraphConv(nn.Module):
    """A graph neural network layer.

    EnGraphConv stands for a Graph Convolution E(n)-equivariant layer. It is the
    equivalent of a linear layer in an MLP, a conv layer in a CNN, or a graph
    conv layer in a GCN.
    """

    def __init__(
            self,
            node_feat,
            edge_feat=0,
            coord_feat=16,
            fourier_feat=0,
            norm_rel_coords=False,
            norm_coord_weights=False,
            num_nearest_neighbors=0,
            dropout=0.0,
            init_eps=1e-3
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
        norm_rel_coords : boolean
            Fourier feature size.
        norm_coord_weights : boolean
            Fourier feature size.
        num_nearest_neighbors : int
            Fourier feature size.
        dropout : float
            Fourier feature size.
        init_eps : float
            Fourier feature size.
        """
        super().__init__()
        self.fourier_feat = fourier_feat

        edge_input_dim = (fourier_feat * 2) + (node_feat * 2) + edge_feat + 1
        dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * 2),
            dropout,
            GELU(),
            nn.Linear(edge_input_dim * 2, coord_feat),
            GELU()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(node_feat + coord_feat, node_feat * 2),
            dropout,
            GELU(),
            nn.Linear(node_feat * 2, node_feat),
        )

        self.norm_coord_weights = norm_coord_weights
        self.norm_rel_coords = norm_rel_coords

        if norm_rel_coords:
            self.rel_coords_scale = nn.Parameter(torch.ones(1))

        self.coords_mlp = nn.Sequential(
            nn.Linear(coord_feat, coord_feat * 4),
            dropout,
            GELU(),
            nn.Linear(coord_feat * 4, 1)
        )

        self.num_nearest_neighbors = num_nearest_neighbors

        self.init_eps = init_eps
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # Seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.normal_(module.weight, std=self.init_eps)

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
        nbhd_indices = None
        b, n, d, fourier_features, num_nearest = *h.shape, self.fourier_feat, self.num_nearest_neighbors
        use_nearest = num_nearest > 0

        rel_coords = rearrange(x, 'b i d -> b i () d') - rearrange(x, 'b j d -> b () j d')
        rel_dist = (rel_coords ** 2).sum(dim=-1, keepdim=True)

        if use_nearest:
            nbhd_indices = rel_dist[..., 0].topk(num_nearest, dim=-1, largest=False).indices
            rel_coords = batched_index_select(rel_coords, nbhd_indices, dim=2)
            rel_dist = batched_index_select(rel_dist, nbhd_indices, dim=2)

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

        coord_weights = self.coords_mlp(m_ij)
        coord_weights = rearrange(coord_weights, 'b i j () -> b i j')

        if self.norm_coord_weights:
            coord_weights = coord_weights.tanh()

        if self.norm_rel_coords:
            rel_coords = normalize(rel_coords, dim=-1) * self.rel_coords_scale

        if mask is not None:
            mask_i = rearrange(mask, 'b i -> b i ()')

            if use_nearest:
                mask_j = batched_index_select(mask, nbhd_indices, dim=1)
            else:
                mask_j = rearrange(mask, 'b j -> b () j')

            mask = mask_i * mask_j
            coord_weights.masked_fill_(~mask, 0.)

            # Free GPU memory
            mask_i.detach()
            mask_j.detach()

        coords_out = einsum('b i j, b i j c -> b i c', coord_weights, rel_coords) + x

        m_i = m_ij.sum(dim=-2)

        node_mlp_input = torch.cat((h, m_i), dim=-1)
        node_out = self.node_mlp(node_mlp_input) + h

        # Free GPU memory
        rel_coords.detach()
        rel_dist.detach()
        feats_i.detach()
        feats_j.detach()
        edge_input.detach()
        m_i.detach()
        m_ij.detach()
        coord_weights.detach()
        node_mlp_input.detach()
        if nbhd_indices is not None:
            nbhd_indices.detach()
        if mask is not None:
            mask.detach()

        return node_out, coords_out

    def __repr__(self):
        return f'GConvEn(structure=h{self.node_feat}-x{self.coord_feat}-e{self.edge_feat})'


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code derived from DMLC (https://github.com/dmlc/dgl/blob/master/examples/pytorch/dagnn/main.py):
# -------------------------------------------------------------------------------------------------------------------------------------

class DAGNNConv(nn.Module):
    def __init__(self,
                 in_dim,
                 k):
        super(DAGNNConv, self).__init__()

        self.s = Parameter(torch.FloatTensor(in_dim, 1))
        self.k = k

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('sigmoid')
        nn.init.xavier_uniform_(self.s, gain=gain)

    def forward(self, graph, feats):
        with graph.local_scope():
            results = [feats]

            degs = graph.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm = norm.to(feats.device).unsqueeze(1)

            for _ in range(self.k):
                feats = feats * norm
                graph.ndata['h'] = feats
                graph.update_all(fn.copy_u('h', 'm'),
                                 fn.sum('m', 'h'))
                feats = graph.ndata['h']
                feats = feats * norm
                results.append(feats)

            H = torch.stack(results, dim=1)
            S = sigmoid(torch.matmul(H, self.s))
            S = S.permute(0, 2, 1)
            H = torch.matmul(S, H).squeeze()

            return H


class MLPLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 bias=True,
                 activation=None,
                 dropout=0):
        super(MLPLayer, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = 1.
        if self.activation is relu:
            gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, feats):
        feats = self.dropout(feats)
        feats = self.linear(feats)
        if self.activation:
            feats = self.activation(feats)

        return feats


class DAGNN(nn.Module):
    def __init__(self,
                 k,
                 in_dim,
                 hid_dim,
                 out_dim,
                 bias=True,
                 activation=relu,
                 dropout=0, ):
        super(DAGNN, self).__init__()
        self.mlp = nn.ModuleList()
        self.mlp.append(MLPLayer(in_dim=in_dim, out_dim=hid_dim, bias=bias,
                                 activation=activation, dropout=dropout))
        self.mlp.append(MLPLayer(in_dim=hid_dim, out_dim=out_dim, bias=bias,
                                 activation=None, dropout=dropout))
        self.dagnn = DAGNNConv(in_dim=out_dim, k=k)

    def forward(self, graph, feats):
        for layer in self.mlp:
            feats = layer(feats)
        feats = self.dagnn(graph, feats)
        return feats


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepInteract (https://github.com/jianlin-cheng/DeepInteract):
# -------------------------------------------------------------------------------------------------------------------------------------
class SAGEConv(nn.Module):
    """GraphSAGE convolution module used by the GraphSAGE model.
    This variant of the SAGEConv layer is able to infer edges via a soft estimation on messages.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """

    def __init__(self, in_feat, out_feat):
        super(SAGEConv, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear = nn.Linear(in_feat * 2, out_feat)

    def forward(self, g, h):
        """Forward computation

        Parameters
        ----------
        g : Graph
            The input graph.
        h : Tensor
            The input node feature.
        """
        with g.local_scope():
            g.ndata['h'] = h
            # update_all is a message passing API.
            g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.mean('m', 'h_N'))
            h_N = g.ndata['h_N']
            h_total = torch.cat([h, h_N], dim=1)
            return self.linear(h_total)
