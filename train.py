import yaml
import argparse
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger

from model import SemanticSegmentationModel
from dataset_module import DatasetModule
from torchlightning_module import torch_lightning_DataModule


# define arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default=None,
                    help="path to the saved checkpoint")
args = parser.parse_args()


with open("hyperparameters.yaml", "r") as stream:
    try:
        hyperparameters = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# set checkpoint path from argument
if args.checkpoint_path is not None:
    hyperparameters["checkpoint_path"] = args.checkpoint_path



# Pass hyperparameters to your PyTorch Lightning model



if __name__=='__main__':

    ## pl components
    model = SemanticSegmentationModel(**hyperparameters["model"])
    data=torch_lightning_DataModule(**hyperparameters["datamodule"])
    wandb_logger=WandbLogger(**hyperparameters["wandb_logger"])

    #create trainer and fit the model
    trainer = pl.Trainer(wandb_logger,**hyperparameters["trainer"])
    trainer.fit(model,data)
