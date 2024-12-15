import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric.datasets as datasets
import torch_geometric
from einops import rearrange
import torch
from tqdm import trange
from att import GraphTransformer
import typer
experiment = typer.Typer(pretty_exceptions_show_locals=False)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        hid_dim = 32
        self.embedder = nn.Linear(1433, hid_dim)
        self.model = GraphTransformer(
            dim=hid_dim,
            depth=1,
            dim_head=8,
            heads=4,
            gated_residual=True,
            with_feedforwards=True,
            ff_mult=4,
            norm_edges=True,
            accept_adjacency_matrix=False,
        )
        self.classifier = nn.Linear(hid_dim, 7)
    
    def forward(self, nodes, edge_index):
        nodes = self.embedder(nodes)
        nodes, edges = self.model(nodes, edge_index)
        return self.classifier(nodes)

    def get_loss(self, nodes, edge_index, labels, train_mask, val_mask):
        logits = self(nodes, edge_index)
        train_logits = logits[train_mask]
        val_logits = logits[val_mask]
        train_labels = labels[train_mask]
        val_labels = labels[val_mask]

        loss = F.cross_entropy(train_logits, train_labels)
        with torch.no_grad():
            train_acc = (train_logits.argmax(-1) == train_labels).count_nonzero() / train_labels.numel()
            val_acc = (val_logits.argmax(-1) == val_labels).count_nonzero() / val_labels.numel()

        return loss, train_acc, val_acc
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 0.01)


def train(model, dataset, max_epochs=1, size=100000):
    optim = model.configure_optimizers()
    nodes = dataset.x[:size]
    edge_index = dataset.edge_index[:, (dataset.edge_index < size).all(0)]
    nodes = nodes.unsqueeze(0)
    labels = dataset.y[:size].unsqueeze(0)
    train_mask = dataset.train_mask[:size].unsqueeze(0)
    val_mask = dataset.val_mask[:size].unsqueeze(0)
    model.train()
    for epoch in range(max_epochs):
        optim.zero_grad()
        loss, train_acc, val_acc = model.get_loss(nodes, edge_index, labels, train_mask, val_mask)
        print(f'{epoch}/{max_epochs} loss={round(loss.item(), 2)} train_acc={round(train_acc.item(), 3)} val_acc={round(val_acc.item(), 3)}')
        loss.backward()
        optim.step()
    model.eval()


@experiment.command()
def main():
    model = Classifier()
    param_size = 0
    with torch.no_grad():
        for name, w in model.named_parameters():
            param_size += w.nelement()
    print(param_size)
    dataset = datasets.Planetoid(
            root="~/data",
            name='Cora',
            split="public",
            transform=torch_geometric.transforms.GCNNorm()
        )

    train(model, dataset, max_epochs=50, size=2708)


if __name__ == '__main__':
    experiment()