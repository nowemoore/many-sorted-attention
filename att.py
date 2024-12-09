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
from torch import nn, einsum
from torch_geometric.nn import Sequential
from einops import rearrange, repeat

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

    def __init__(self, dim, dim_head=64, heads=8, edge_dim=None):
        super().__init__()
        # fundamental change: consider edges separately
        # TODO argument for dropping edges at the final layer such that we don't compute edge values their values for node prediction
        edge_dim = default(edge_dim, 2 * dim)
        assert edge_dim >= 2 * dim, 'edges must contain both adjacent node features: so edge_dim >= 2 * dim'

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.e_to_q = nn.Linear(edge_dim, inner_dim)
        self.e_to_k = nn.Linear(edge_dim, inner_dim)
        self.e_to_v = nn.Linear(edge_dim, inner_dim)

        self.to_out = nn.Linear(inner_dim, dim)
        self.e_to_out = nn.Linear(inner_dim, edge_dim)

    def forward(self, nodes, edges, mask=None):
        """
        nodes: array[batch, nodes, dim]
        edges: array[batch, edges, edge_dim]
        """
        h = self.heads

        q = self.to_q(nodes)
        k = self.to_k(nodes)
        v = self.to_v(nodes)

        e_q = self.e_to_q(edges)
        e_k = self.e_to_k(edges)
        e_v = self.e_to_v(edges)

        q, k, v, e_q, e_k, e_v = map(
            lambda t: rearrange(t, 'b ... (h d) -> (b h) ... d', h=h),
            (q, k, v, e_q, e_k, e_v))

        k, v, e_k, e_v = map(lambda t: rearrange(t, 'b j d -> b () j d '),
                             (k, v, e_k, e_v))

        q = torch.cat((q, e_q), dim=1)
        k = torch.cat((k, e_k), dim=2)
        v = torch.cat((v, e_v), dim=2)

        sim = einsum('b i d, b i j d -> b i j', q, k) * self.scale

        # TODO mask will have incorrect dimensionality
        if exists(mask):
            mask = rearrange(mask, 'b i -> b i ()') & rearrange(
                mask, 'b j -> b () j')
            mask = repeat(mask, 'b i j -> (b h) i j', h=h)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b i j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out, e_out = out.split([nodes.size(1), edges.size(1)], dim=1)
        return self.to_out(out), self.e_to_out(e_out)


# optional feedforward


def FeedForward(dim, ff_mult=4):
    return nn.Sequential(nn.Linear(dim, dim * ff_mult), nn.GELU(),
                         nn.Linear(dim * ff_mult, dim))


# classes


class GraphTransformer(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 dim_head=64,
                 edge_dim=None,
                 heads=8,
                 gated_residual=True,
                 with_feedforwards=False,
                 norm_edges=False,
                 accept_adjacency_matrix=False):
        super().__init__()
        self.layers = List([])
        edge_dim = default(edge_dim, 2 * dim)
        assert edge_dim >= 2 * dim, 'edges must contain both adjacent node features: so edge_dim >= 2 * dim'
        self.norm_edges = nn.LayerNorm(
            edge_dim) if norm_edges else nn.Identity()

        for _ in range(depth):
            self.layers.append(
                List([
                    Sequential(
                        'n0, e0',
                        [(nn.LayerNorm(dim), 'n0 -> n'),
                         (nn.LayerNorm(edge_dim), 'e0 -> e'),
                         (Attention(dim,
                                    edge_dim=edge_dim,
                                    dim_head=dim_head,
                                    heads=heads), 'n, e -> n, e'),
                         (GatedResidual(dim), 'n, n0 -> n'),
                         (GatedResidual(edge_dim), 'e, e0 -> e'),
                         (lambda n, e: (n, e), 'n, e -> n, e')],
                    ),
                    Sequential('n, e', [(nn.LayerNorm(dim), 'n0 -> n'),
                                        (FeedForward(dim), 'n -> n'),
                                        (GatedResidual(dim), 'n, n0 -> n'),
                                        (nn.LayerNorm(dim), 'e0 -> e'),
                                        (FeedForward(dim), 'e -> e'),
                                        (GatedResidual(dim), 'e, e0 -> e'),
                                        (lambda n, e: (n, e), 'n, e -> n, e')])
                    if with_feedforwards else None
                ]))

    def forward(self, nodes, edges=None, adj_mat=None, mask=None):
        batch, seq, _ = nodes.shape

        if exists(adj_mat):
            assert not exists(edges)
            assert adj_mat.shape == (batch, seq, seq)
            # edges = {[n(u), n(v)] | adj_mat[u, v] = 1}
            edges = nodes[torch.argwhere(adj_matrix)].flatten(-2, -1)

        edges = self.norm_edges(edges)

        for attn_block, ff_block in self.layers:
            nodes, edges = attn_block(nodes, edges)

            if exists(ff_block):
                nodes, edges = ff_block(nodes, edges)

        return nodes, edges
