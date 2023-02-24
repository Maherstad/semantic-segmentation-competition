import torch
import numpy as np
import segmentation_models_pytorch as smp

import torch.nn as nn
from torchmetrics.functional import accuracy

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from torchlightning_module import torch_lightning_DataModule
from dataset_module import DatasetModule

#torch.set_float32_matmul_precision('medium')

# create a device object explicitly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###
class SemanticSegmentationModel(pl.LightningModule):
    def __init__(self,encoder_name,classes,in_channels,activation,encoder_weights,lr):
        super().__init__()
        self.encoder_name=encoder_name
        self.classes=classes
        self.in_channels=in_channels
        self.activation=activation
        self.encoder_weights=encoder_weights
        self.lr=lr
        self.loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters()
        self.model = smp.Unet(
           encoder_name=self.encoder_name,
           classes=self.classes,
           activation=self.activation,
           encoder_weights=self.encoder_weights,
            in_channels=self.in_channels,
              )
        #self.lr=self.hparams['lr']

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        labels = labels.squeeze(1)

        class_weights = torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0], dtype=torch.float32, device='cuda')
        self.loss_fn.weight = class_weights

        loss = self.loss_fn(outputs, labels)
        self.log('train_loss',loss, on_step=False, on_epoch=True)
        return {'loss': loss}

#     def training_epoch_end(self, outputs):
#         avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
#         self.log("avg_train_loss", avg_loss)
#         wandb.log({"avg_train_loss": avg_loss})


    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        labels = labels.squeeze(1)
        class_weights = torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0], dtype=torch.float32, device='cuda')
        self.loss_fn.weight = class_weights


        loss = self.loss_fn(outputs, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return {'val_loss': loss}

#     def validation_epoch_end(self, outputs):
#         avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
#         logs = {'val_loss': avg_loss}
#         return {'val_loss': avg_loss, 'log': logs}

        # def validation_epoch_end(self, outputs):
        #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        #     logs = {'val_loss': avg_loss}
        #     self.log('val_loss', avg_loss)



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.1,
                                                               patience=3,
                                                               verbose=True)
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min')
        checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints',
                                               filename='model-{epoch:02d}-{val_loss:.2f}',
                                               monitor='val_loss',
                                               save_top_k=3,
                                               mode='min')
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
            'callbacks': [early_stop_callback, checkpoint_callback]
        }

# def lr_schedule(step):
#    lr = 0.001
#    if step < 10:
#        return lr
#    elif step < 20:
#        return lr / 2
#    else:
#        return lr / 4

#lr_scheduler = pl.callbacks.LearningRateScheduler(lr_schedule)
