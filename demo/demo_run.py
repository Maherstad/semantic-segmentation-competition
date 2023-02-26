#general
import os
import numpy as np
import json
import random
from pathlib import Path

#deep learning
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning import Trainer, seed_everything
try:
  from pytorch_lightning.utilities.distributed import rank_zero_only
except ImportError:
  from pytorch_lightning.utilities.rank_zero import rank_zero_only

import albumentations as A

#flair-one baseline modules
from py_module.utils import load_data, subset_debug
from py_module.datamodule import OCS_DataModule
from py_module.model import SMP_Unet_meta
from py_module.task_module import SegmentationTask
from py_module.writer import PredictionWriter


if __name__ == '__main__':

    ##############################################################################################
    # paths and naming
    path_data = "/Users/alsadeq/Documents/semantic-segmentation-and-domain-adaptation/semantic-segmentation-competition/raw_dataset" # toy (or full) dataset folder
    path_metadata_file = "./metadata/flair-one_TOY_metadata.json" # json file containing the metadata

    out_folder = "/content" # output directory for logs and predictions.
    out_model_name = "flair-one-baseline_argu" # to keep track
    ##############################################################################################

    ##############################################################################################
    # tasking
    use_weights = False
    class_weights = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0]

    use_metadata = False
    use_augmentation = False
    ##############################################################################################

    ##############################################################################################
    # training hyper-parameters
    batch_size = 2
    learning_rate = 0.001
    num_epochs = 10
    ##############################################################################################

    ##############################################################################################
    # computational ressources
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu' # set to 'cpu' if GPU not available
    gpus_per_node = 1 # set to 1 if mono-GPU
    num_nodes = 1 # set to 1 if mono-GPU
    strategy = None # Put this parameter to None if train on only one GPU or on CPUs. If multiple GPU, set to 'ddp'
    num_workers = 1
    ##############################################################################################

    ##############################################################################################
    # display
    enable_progress_bar = True
    progress_rate = 10 #tqdm update rate during training
    ##############################################################################################

    #out_dir = Path(out_folder, out_model_name)
    #out_dir.mkdir(parents=True, exist_ok=True)
    out_dir='./content/flair-one-baseline_argu'

    seed_everything(2022, workers=True)

    @rank_zero_only
    def step_loading(path_data, path_metadata_file: str, use_metadata: bool) -> dict:
        print('+'+'-'*29+'+', '   LOADING DATA   ', '+'+'-'*29+'+')
        train, val, test = load_data(path_data, path_metadata_file, use_metadata=use_metadata)
        return train, val, test


    @rank_zero_only
    def print_recap():
        print('\n+'+'='*80+'+',f"{'Model name: '+out_model_name : ^80}", '+'+'='*80+'+', f"{'[---TASKING---]'}", sep='\n')
        for info, val in zip(["use weights", "use metadata", "use augmentation"], [use_weights, use_metadata, use_augmentation]): print(f"- {info:25s}: {'':3s}{val}")
        print('\n+'+'-'*80+'+', f"{'[---DATA SPLIT---]'}", sep='\n')
        for split_name, d in zip(["train", "val", "test"], [dict_train, dict_val, dict_test]): print(f"- {split_name:25s}: {'':3s}{len(d['IMG'])} samples")
        print('\n+'+'-'*80+'+', f"{'[---HYPER-PARAMETERS---]'}", sep='\n')
        for info, val in zip(["batch size", "learning rate", "epochs", "nodes", "GPU per nodes", "accelerator", "workers"], [batch_size, learning_rate, num_epochs, num_nodes, gpus_per_node, accelerator, num_workers]): print(f"- {info:25s}: {'':3s}{val}")
        print('\n+'+'-'*80+'+', '\n')

    dict_train, dict_val, dict_test = step_loading(path_data, path_metadata_file, use_metadata=use_metadata)
    print_recap()



    if use_augmentation == True:
        transform_set = A.Compose([
                                    A.VerticalFlip(p=0.5),
                                    A.HorizontalFlip(p=0.5),
                                    A.RandomRotate90(p=0.5)])
    else:
        transform_set = None

    dm = OCS_DataModule(
        dict_train=dict_train,
        dict_val=dict_val,
        dict_test=dict_test,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        num_classes=13,
        num_channels=5,
        use_metadata=use_metadata,
        use_augmentations=transform_set)



    model = SMP_Unet_meta(n_channels=5, n_classes=13, use_metadata=use_metadata)



    if use_weights == True:
        with torch.no_grad():
            class_weights = torch.FloatTensor(class_weights)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        cooldown=4,
        min_lr=1e-7,
    )

    seg_module = SegmentationTask(
        model=model,
        num_classes=dm.num_classes,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        use_metadata=use_metadata
    )

    ckpt_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(out_dir,"checkpoints"),
        filename="ckpt-{epoch:02d}-{val_loss:.2f}"+'_'+out_model_name,
        save_top_k=1,
        mode="min",
        save_weights_only=True, # can be changed accordingly
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=30, # if no improvement after 30 epoch, stop learning.
        mode="min",
    )

    prog_rate = TQDMProgressBar(refresh_rate=progress_rate)

    callbacks = [
        ckpt_callback,
        early_stop_callback,
        prog_rate,
    ]


    logger = TensorBoardLogger(
        save_dir=out_dir,
        name=Path("tensorboard_logs"+'_'+out_model_name).as_posix()
    )

    loggers = [
        logger
    ]

    #### instanciation of  Trainer
    trainer = Trainer(
        accelerator=accelerator,
        devices=gpus_per_node,
        strategy=strategy,
        num_nodes=num_nodes,
        max_epochs=num_epochs,
        num_sanity_val_steps=0,
        callbacks = callbacks,
        logger=loggers,
        enable_progress_bar = enable_progress_bar,
    )


    trainer.fit(seg_module, datamodule=dm)
