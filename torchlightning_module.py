import pandas as pd
from sklearn.model_selection import train_test_split
from dataset_module import DatasetModule
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import random_split


# dataset_path='dataset'
# metadata_path='flair-one_metadata.json'
# img_ids='img_ids.jsonl'
# train_val_balanced_split='val'
# train=True


class torch_lightning_DataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir: str = 'dataset',
                 #metadata:str='flair-one_metadata.json',
                 batch_size: int = 16,
                 num_workers:int=1,
                 pin_memory: bool= True
                ):
        super().__init__()
        self.data_dir = data_dir
        #self.metadata=metadata
        self.batch_size = batch_size
        self.num_workers=num_workers
        self.pin_memory=pin_memory
    
    def setup(self, stage: str):
        #dataset, metadata, train
        self.dataset_test = DatasetModule(self.data_dir, train='test') 
        self.dataset_predict = DatasetModule(self.data_dir,train='test')
        self.dataset_train=DatasetModule(self.data_dir,train='train')
        self.dataset_val =DatasetModule(self.data_dir,train='val')



    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size,num_workers=self.num_workers,pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size,num_workers=self.num_workers,pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size,num_workers=self.num_workers,pin_memory=self.pin_memory)

    def predict_dataloader(self):
        return DataLoader(self.dataset_predict,batch_size=self.batch_size,num_workers=self.num_workers,pin_memory=self.pin_memory)

    # def teardown(self, stage: str):
    #     # Used to clean-up when the run is finished
    #     ...