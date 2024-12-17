"""
Implementation of the many-sorted attention layer

This layer uses a shared embedding space, i.e.
(N, E) -> (Q, K, V) -> (N', E')
as opposed to
(N, E) -> ((Q_N, K_N, V_N), (Q_E, K_E, V_E)) -> (N', E')

This code was heavily adapted from Graph Transformer
https://github.com/lucidrains/graph-transformer-pytorch
"""
import torch
from torch import nn
from torch_geometric.nn import Sequential
from einops import rearrange, repeat, einsum

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


List = nn.ModuleList


# gated residual
class GatedResidual(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim * 3, 1, bias=False),
                                  nn.Sigmoid())

    def forward(self, x, res):
        gate_input = torch.cat((x, res, x - res), dim=-1)
        gate = self.proj(gate_input)
        return x * gate + res * (1 - gate)


# attention


class Attention(nn.Module):

    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        edge_dim=None,
        edge_out_dim=None,
        alpha=0.8,
        out_edges=True,
    ):
        super().__init__()
        # fundamental change: consider edges separately
        # TODO argument for dropping edges at the final layer such that we don't compute edge values their values for node prediction
        edge_dim = default(edge_dim, 2 * dim)
        edge_out_dim = default(edge_out_dim, dim)
        assert edge_dim >= 2 * dim, 'edges must contain both adjacent node features: so edge_dim >= 2 * dim'
        self.out_edges = out_edges

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        # node qkv
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)

        # edge qkv, separated for implementation of prior
        self.e_to_q_uv = nn.Linear(2 * dim, inner_dim)
        self.e_to_k_uv = nn.Linear(2 * dim, inner_dim)
        self.e_to_v_uv = nn.Linear(2 * dim, inner_dim)
        self.e_to_q_w = nn.Linear(edge_dim - 2 * dim, inner_dim)
        self.e_to_k_w = nn.Linear(edge_dim - 2 * dim, inner_dim)
        self.e_to_v_w = nn.Linear(edge_dim - 2 * dim, inner_dim)

        self.to_out = nn.Linear(inner_dim, dim)
        if self.out_edges:
            self.e_to_out = nn.Linear(inner_dim, edge_out_dim)
        self.alpha = 0.8

    def prepare_edges(self, nodes, edge_index):
        # TODO deal with batching properly
        # TODO deal with edge_feature dimensionality possibly being different at each layer (i.e. being 0 in layer 0) but nonzero later
        # TODO option to not output edges in the final layer
        edges = rearrange(
            nodes.view(nodes.size()[1:])[edge_index], 'n e d -> 1 e (n d)')
        return edges

    def forward(self, nodes, edge_features, edge_index, mask=None):
        """
        nodes: array[batch, nodes, dim]
        edges: array[batch, edges, edge_dim]
        """
        h = self.heads

        q = self.to_q(nodes)
        k = self.to_k(nodes)
        v = self.to_v(nodes)

        edges = self.prepare_edges(nodes, edge_index)

        # impose prior to aid learning
        e_q_uv = self.e_to_q_uv(edges)
        e_k_uv = self.e_to_k_uv(edges)
        e_v_uv = self.e_to_v_uv(edges)

        # TODO which way around should it be? Do we just want to impose a restriction on the key? on the query? the value? all 3?

        e_q = (1 - self.alpha) * 2**0.5 * e_q_uv + self.alpha * einsum(
            k[:, edge_index], 'b e n d -> b n d')
        e_k = (1 - self.alpha) * 2**0.5 * e_k_uv + self.alpha * einsum(
            q[:, edge_index], 'b e n d -> b n d')
        e_v = (1 - self.alpha) * 2**0.5 * e_v_uv + self.alpha * einsum(
            v[:, edge_index], 'b e n d -> b n d')

        if exists(edge_features):
            e_q = e_q + self.e_to_k_w(edge_features)
            e_k = e_k + self.e_to_v_w(edge_features)
            e_v = e_v + self.e_to_q_w(edge_features)

        # normalise to preserve magnitude of input signals
        e_q *= nodes.size(-1)**0.5 / edges.size(-1)**0.5
        e_k *= nodes.size(-1)**0.5 / edges.size(-1)**0.5
        e_v *= nodes.size(-1)**0.5 / edges.size(-1)**0.5

        q, k, v, e_q, e_k, e_v = map(
            lambda t: rearrange(t, 'b ... (h d) -> (b h) ... d', h=h),
            (q, k, v, e_q, e_k, e_v))

        k, v, e_k, e_v = map(lambda t: rearrange(t, 'b j d -> b () j d '),
                             (k, v, e_k, e_v))

        q = torch.cat((q, e_q), dim=1)
        k = torch.cat((k, e_k), dim=2)
        v = torch.cat((v, e_v), dim=2)

        sim = einsum(q, k, 'b i d, b i j d -> b i j') * self.scale

        # TODO mask will have incorrect dimensionality
        if exists(mask):
            mask = rearrange(mask, 'b i -> b i ()') & rearrange(
                mask, 'b j -> b () j')
            mask = repeat(mask, 'b i j -> (b h) i j', h=h)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        out = einsum(attn, v, 'b i j, b i j d -> b i d')
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out, e_out = out.split([nodes.size(1), edges.size(1)], dim=1)
        if self.out_edges:
            return self.to_out(out), self.e_to_out(e_out)
        else:
            return self.to_out(out), None


# optional feedforward


def FeedForward(dim, ff_mult=4):
    return nn.Sequential(nn.Linear(dim, dim * ff_mult), nn.GELU(),
                         nn.Linear(dim * ff_mult, dim))


# classes


class TypedTransformer(nn.Module):

    def __init__(self,
                 dim,
                 hid_dim,
                 depth,
                 dim_head=64,
                 edge_dim=None,
                 heads=8,
                 gated_residual=True,
                 with_feedforwards=False,
                 norm_edges=False,
                 ff_mult=4,
                 accept_adjacency_matrix=False,
                 out_dim=None,
                 alpha=0.8,
                 agg=None):
        super().__init__()
        self.layers = List([])
        edge_dim = default(edge_dim, 0)
        out_dim = default(out_dim, hid_dim)
        assert agg in {None, 'mean'}
        self.agg = agg

        self.encoder = nn.Linear(dim, hid_dim)

        for layer in range(depth):
            self.layers.append(
                List([
                    Sequential(
                        'n_f, e_f, idx',
                        [(nn.LayerNorm(hid_dim), 'n_f -> n'),
                         (nn.LayerNorm(edge_dim)
                          if edge_dim else lambda e_f: e_f, 'e_f -> e'),
                         (Attention(hid_dim,
                                    edge_dim=edge_dim + 2 * hid_dim,
                                    edge_out_dim=edge_dim,
                                    dim_head=dim_head,
                                    heads=heads,
                                    alpha=alpha,
                                    out_edges=layer != depth - 1),
                          ('n, e, idx -> n, e')),
                         (GatedResidual(hid_dim), 'n, n_f -> n'),
                         (GatedResidual(edge_dim) if edge_dim
                          and layer != depth - 1 else lambda e, e_f: e,
                          'e, e_f -> e'), (lambda n, e:
                                           (n, e), 'n, e -> n, e')],
                    ),
                    Sequential('n0, e0', [
                        (nn.LayerNorm(hid_dim), 'n0 -> n'),
                        (FeedForward(hid_dim, ff_mult), 'n -> n'),
                        (GatedResidual(hid_dim), 'n, n0 -> n'),
                        *(((nn.LayerNorm(edge_dim)
                            if norm_edges else lambda x: x, 'e0 -> e'),
                           (FeedForward(edge_dim, ff_mult), 'e -> e'),
                           (GatedResidual(edge_dim), 'e, e0 -> e'),
                           (lambda n, e:
                            (n, e), 'n, e -> n, e')) if layer != depth - 1 else
                          ((lambda n, e: (n, e), 'n, e0 -> n, e0'), ))
                    ]) if with_feedforwards else None
                ]))

        self.classifier = nn.Linear(hid_dim, out_dim)

    def forward(self, nodes, edge_index, edge_features=None, mask=None):
        # nodes = rearrange(nodes, 'n d -> 1 n d')
        nodes = self.encoder(nodes)
        if len(nodes.size()) == 2:
            nodes = nodes.unsqueeze(0)
        if exists(edge_features):
            edge_features = rearrange(edge_features, 'e d -> 1 e d')

        for attn_block, ff_block in self.layers:
            nodes, edge_features = attn_block(nodes, edge_features, edge_index)

            if exists(ff_block):
                # Note: in the last layer ff_block computes identity for edge_features
                nodes, edge_features = ff_block(nodes, edge_features)

        match self.agg:
            case None:  # for node classification
                return self.classifier(nodes)
            case 'mean':  # for graph classification
                return self.classifier(nodes.mean(-2))
            case _:
                raise NotImplementedError(
                    'This aggregation method is not implemented')
