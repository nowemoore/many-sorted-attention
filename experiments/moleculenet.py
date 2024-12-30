'''
This tests the performance of edge set attention on MoleculeNet
'''
import torch
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

    def __init__(self, model, params, optparams, max_epochs):
        super().__init__()
        self.model = model(**params)
        self.optparams = optparams
        self.max_epochs = max_epochs

    def forward(self, edge_idx, x, batch, ptr):
        return self.model(edge_idx, x, batch, ptr)

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(self.parameters(), **self.optparams)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.max_epochs)
        return [optimiser], [scheduler]

    def training_step(self, batch, batch_idx):
        edge_idx, x, y, batch, ptr = tuple(x[1] for x in batch)

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
        'dim': 6,
        'depth': 4,
        'dim_head': 4,
        'heads': 4,
        'k': 1,
        'out_dim': 6,
        'gated_residual': False
    }, {
        'lr': 0.0001,
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
        accelerator='cpu',
        logger=logger,
        gradient_clip_val=0.5,
        gradient_clip_algorithm='value',
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    main('esa', max_epochs=100)
