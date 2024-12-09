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
        self.model = GraphTransformer(
            dim=1433,
            depth=1,
            dim_head=8,
            heads=8,
            gated_residual=True,
            with_feedforwards=True,
            norm_edges=False,
            accept_adjacency_matrix=False,
        )
        self.classifier = nn.Linear(1433, 7)
    
    def forward(self, nodes, edges):
        nodes, edges = self.model(nodes, edges)
        return self.classifier(nodes)

    def get_loss(self, nodes, edges, labels, mask):
        logits = self(nodes, edges)[mask]
        labels = labels[mask]
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(-1) == labels).count_nonzero() / labels.numel()
        return loss, accuracy
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 0.01, weight_decay=0.1)


def train(model, dataset, max_epochs=1):
    optim = model.configure_optimizers()
    nodes = dataset.x
    edges = rearrange(nodes[dataset.edge_index], 'n e d -> 1 e (n d)')
    nodes = nodes.unsqueeze(0)
    labels = dataset.y.unsqueeze(0)
    mask = dataset.train_mask.unsqueeze(0)
    model.train()
    for epoch in trange(max_epochs):
        optim.zero_grad()
        loss, acc = model.get_loss(nodes, edges, labels, mask)
        loss.backward()
        optim.step()
    model.eval()


@experiment.command()
def main():
    model = Classifier()
    dataset = datasets.Planetoid(
            root="~/data",
            name='Cora',
            split="public",
            transform=torch_geometric.transforms.GCNNorm()
        )

    train(model, dataset, max_epochs=1)


if __name__ == '__main__':
    experiment()