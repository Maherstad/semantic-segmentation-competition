import yaml
from pytorch_lightning.callbacks import EarlyStopping #, ModelCheckpoint
from model import SemanticSegmentationModel
from pytorch_lightning.loggers import WandbLogger
from torchlightning_module import torch_lightning_DataModule
from dataset_module import DatasetModule
import pytorch_lightning as pl


with open("hyperparameters.yaml", "r") as stream:
    try:
        hyperparameters = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=500,
    mode='min'
)
        
        
# Pass hyperparameters to your PyTorch Lightning model

if __name__=='__main__':
    model = SemanticSegmentationModel(**hyperparameters["model"])
    data=torch_lightning_DataModule(**hyperparameters["datamodule"])
    wandb_logger=WandbLogger(**hyperparameters["wandb_logger"])
    trainer = pl.Trainer(wandb_logger,**hyperparameters["trainer"])
    trainer.fit(model,data)
    trainer.save_checkpoint('model.ckpt')
    #trainer.fit(model,data, ckpt_path="model.ckpt")

