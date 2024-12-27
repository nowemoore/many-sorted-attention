from models.esa import *
from experiments.enzymes import *


if __name__ == '__main__':
    datamodule = DataModule()
    datamodule.setup('fit')
    dataloader = datamodule.train_dataloader()
    data = next(iter(dataloader))

    model = Model(*models['esa'], max_epochs=1)
    model.training_step(data, 0)
