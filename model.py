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
        #print('XXXXXXXXXXXXXXXX',encoder_name)
        #print('XXXXXXXXXXXXXXXXZZZ',self.encoder_weights)

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
        loss = self.loss_fn(outputs, labels)
        
        
        #outputs : output of model > logits /// labels > y ///images > x 
        self.log('train_loss',loss, on_step=False, on_epoch=True)
        return loss
    
    
#     def training_epoch_end(self, outputs):
#         avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
#         self.log("avg_train_loss", avg_loss)
#         wandb.log({"avg_train_loss": avg_loss})


    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        labels = labels.squeeze(1)
        #print(f'{labels.shape} is shape of label in the val step method')
        #print(f'{outputs.shape} is shape of outputs in the val step method')

        loss = self.loss_fn(outputs, labels)
        # Log loss and accuracy
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
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    
# def lr_schedule(step):
#    lr = 0.001
#    if step < 10:
#        return lr
#    elif step < 20:
#        return lr / 2
#    else:
#        return lr / 4

#lr_scheduler = pl.callbacks.LearningRateScheduler(lr_schedule)
