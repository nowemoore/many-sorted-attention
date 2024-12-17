import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.datasets as datasets
import torch_geometric
from einops import rearrange
import torch
from tqdm import trange
import optuna
from att import TypedTransformer
import typer

experiment = typer.Typer(pretty_exceptions_show_locals=False)


class Classifier(nn.Module):

    def __init__(self,
                 hid_dim=16,
                 out_dim=7,
                 depth=3,
                 dim_head=8,
                 heads=4,
                 alpha=0.9,
                 ff_mult=4):
        super().__init__()
        self.embedder = nn.Linear(1433, hid_dim)
        self.model = TypedTransformer(
            dim=hid_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            gated_residual=True,
            with_feedforwards=False,
            ff_mult=ff_mult,
            norm_edges=True,
            accept_adjacency_matrix=False,
            alpha=alpha,
        )

    def forward(self, nodes, edge_index):
        nodes = self.embedder(nodes)
        return self.model(nodes, edge_index)

    def get_loss(self, nodes, edge_index, labels, train_mask, val_mask):
        logits = self(nodes, edge_index)
        train_logits = logits[train_mask]
        val_logits = logits[val_mask]
        train_labels = labels[train_mask]
        val_labels = labels[val_mask]

        loss = F.cross_entropy(train_logits, train_labels)
        with torch.no_grad():
            train_acc = (train_logits.argmax(-1) == train_labels
                         ).count_nonzero() / train_labels.numel()
            val_acc = (val_logits.argmax(-1)
                       == val_labels).count_nonzero() / val_labels.numel()

        return loss, train_acc, val_acc

    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.parameters(),
                                     lr=0.02,
                                     weight_decay=0.00001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimiser, T_max=50)
        return optimiser, scheduler


def train(model, dataset, max_epochs=1, size=100000, log=True):
    optim, scheduler = model.configure_optimizers()
    nodes = dataset.x[:size]
    edge_index = dataset.edge_index[:, (dataset.edge_index < size).all(0)]
    nodes = nodes.unsqueeze(0)
    labels = dataset.y[:size].unsqueeze(0)
    train_mask = dataset.train_mask[:size].unsqueeze(0)
    val_mask = dataset.val_mask[:size].unsqueeze(0)
    model.train()
    val_accs = []
    for epoch in range(max_epochs):
        optim.zero_grad()
        loss, train_acc, val_acc = model.get_loss(nodes, edge_index, labels,
                                                  train_mask, val_mask)
        if log is True:
            print(
                f'{epoch}/{max_epochs} loss={round(loss.item(), 2)} train_acc={round(train_acc.item(), 3)} val_acc={round(val_acc.item(), 3)}'
            )
        val_accs.append(round(val_acc.item(), 3))
        loss.backward()
        optim.step()
        scheduler.step()
    model.eval()

    return max(val_accs)


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
        transform=torch_geometric.transforms.GCNNorm())

    train(model, dataset, max_epochs=50, size=2000)


dataset = datasets.Planetoid(root="~/data",
                             name='Cora',
                             split="public",
                             transform=torch_geometric.transforms.GCNNorm())


def objective(trial):
    hid_dim = trial.suggest_int('hid_dim', 22, 32)
    depth = trial.suggest_int('depth', 2, 3)
    dim_head = trial.suggest_int('dim_head', 8, 16)
    heads = trial.suggest_int('heads', 2, 8)
    # ff_mult = trial.suggest_int('ff_mult', 1, 2)
    alpha = trial.suggest_float('alpha', 0.0, 0.5)

    model = Classifier(
        hid_dim=hid_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        # ff_mult=ff_mult,
        alpha=alpha,
    )
    return train(model, dataset, max_epochs=50, size=2000, log=False)


def hyperparam_search():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=3)
    print(f'best_params: {study.best_params}')
    print(f'importances: {optuna.importance.get_param_importances(study)}')


@experiment.command()
def test():
    model = Classifier(
        hid_dim=26,
        depth=2,
        dim_head=10,
        heads=8,
        # ff_mult=ff_mult,
        alpha=0.2,
    )
    return train(model, dataset, max_epochs=50, size=2000)


if __name__ == '__main__':
    experiment()
