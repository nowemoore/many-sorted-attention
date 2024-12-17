'''
This tests the performance of typed attention on Enzymes
'''
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from att import TypedTransformer
import typer

experiment = typer.Typer(pretty_exceptions_show_locals=False)


def train(model, dataloader, max_epochs=50, log=True):
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=0.01,
                                 weight_decay=1e-6)

    model.train()
    for epoch in range(max_epochs):
        losses = []
        accs = []
        for i, batch in enumerate(dataloader):
            # minibatching for training purposes, not performance
            with torch.no_grad():
                indices = batch.batch
                xs = [
                    batch.x[indices == i]
                    for i in range(batch.ptr.size(0) - 1)
                ]
                ys = batch.y
                edge_indices = [
                    batch.edge_index[:, indices[batch.edge_index[0]] == i]
                    for i in range(batch.ptr.size(0) - 1)
                ]

            optimiser.zero_grad()

            loss = 0
            acc = 0
            for j, (x, y, edge_index) in enumerate(zip(xs, ys.unsqueeze(1), edge_indices)):
                logits = model(nodes=x,
                               edge_index=edge_index - batch.ptr[j] - (0 if j == 0 else 1),
                               edge_features=torch.zeros(
                                   edge_index.size(1), 0))
                loss += F.cross_entropy(logits, y) / ys.size(0)
                acc += (logits.argmax(-1)
                        == y).count_nonzero() / ys.size(0)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
            accs.append(acc.item())
        if log:
            print(f'{epoch=}/{max_epochs} loss={round(sum(losses) / len(losses), 3)} acc={round(sum(accs) / len(accs), 3)}')


@experiment.command()
def main():
    dataset = TUDataset(root='~/data', name='ENZYMES')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TypedTransformer(
        dim=3,
        hid_dim=6,
        out_dim=6,
        depth=1,
        dim_head=8,
        edge_dim=0,
        heads=4,
        gated_residual=True,
        with_feedforwards=True,
        norm_edges=False,
        ff_mult=4,
        alpha=0.4,
        agg='mean'
    )
    train(model, dataloader)


if __name__ == '__main__':
    experiment()
