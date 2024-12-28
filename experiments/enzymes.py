'''
This tests the performance of typed attention on Enzymes
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from models import ESATransformer
from einops import rearrange
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb


class Model(pl.LightningModule):

    def __init__(self, model, params, optparams, max_epochs, effective_batch=128):
        super().__init__()
        embedding = nn.Embedding(9, params['dim'])
        self.model = model(**params, embedding=embedding)
        self.optparams = optparams
        self.max_epochs = max_epochs
        self.effective_batch = effective_batch

    def forward(self, edge_idx, x, batch, ptr):
        return self.model(edge_idx, x, batch, ptr)

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(self.parameters(), **self.optparams)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.max_epochs)
        return [optimiser], [scheduler]

    def training_step(self, batch, batch_idx):
        loss = 0
        acc = 0

        if self.effective_batch != None:
            for graph in batch[:]:
                edge_idx = graph.edge_index
                x = graph.x
                x = x.argmax(-1)
                y = graph.y
                batch_ = torch.zeros_like(x)
                ptr = torch.tensor([0, graph.x.size(0)])

                logits = self(edge_idx, x, batch_[edge_idx[0]], ptr)
                loss += F.cross_entropy(logits, y) / batch.y.size(0)
                self.log('train loss', loss)
                acc += (logits.argmax(-1) == y).count_nonzero() / batch.y.size(0)
                self.log('train acc', acc)
        else:
            edge_idx, x, y, batch, ptr = tuple(x[1] for x in batch)
            with torch.no_grad():
                x = x.argmax(-1)

            logits = self(edge_idx, x, batch[edge_idx[0]], ptr)
            loss = F.cross_entropy(logits, y)
            self.log('train loss', loss)
            acc = (logits.argmax(-1) == y).count_nonzero() / y.size(0)
            self.log('train acc', acc)

        return loss


class DataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.dataset = TUDataset(root='~/data', name='ENZYMES')

    def setup(self, stage: str):
        self.train_loader = DataLoader(self.dataset,
                                       batch_size=8,
                                       shuffle=True,
                                       num_workers=16)

    def train_dataloader(self):
        return self.train_loader


models = {
    'esa': (ESATransformer, {
        'dim': 64,
        'layers': ['sab', 'sab', 'mab', 'mab', 'sab', 'sab'],
        'dim_head': 8,
        'heads': 8,
        'k': 8,
        'out_dim': 6,
        'gated_residual': False
    }, {
        'lr': 0.0001,
        'weight_decay': 0.001
    })
}


def main(model: str, max_epochs: int):
    logger = WandbLogger(name=f'{model}-enzymes', project='TypedAttention')
    modelfn, params, optparams = models[model]
    model = Model(modelfn, params, optparams, max_epochs)
    datamodule = DataModule()
    early_stop_callback = EarlyStopping(monitor="val loss",
                                        min_delta=0.00,
                                        patience=100,
                                        verbose=False,
                                        mode="min")

    logger.log_hyperparams(params)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        max_time='00:00:30:00',
        enable_checkpointing=False,
        log_every_n_steps=5,
        accelerator='cpu',
        logger=logger,
        gradient_clip_val=0.5,
        gradient_clip_algorithm='value',
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    main('esa', max_epochs=100)
