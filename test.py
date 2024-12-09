from att import *

if __name__ == '__main__':
    attention = Attention(
        dim=8,
        dim_head=4,
        heads=2,
        edge_dim=18,  # edges = (u|v|w) where w are edge features
    )

    attention(nodes=torch.randn(3, 10, 8),
              edges=torch.randn(3, 23, 18),
              mask=None)

    model = GraphTransformer(
        dim=8,
        edge_dim=18,
        depth=1,
        dim_head=4,
        heads=2,
        gated_residual=False,
        with_feedforwards=False,
        norm_edges=False,
        accept_adjacency_matrix=False,
    )

    nodes, edges = model(
        torch.randn(3, 10, 8),
        torch.randn(3, 23, 18),
    )