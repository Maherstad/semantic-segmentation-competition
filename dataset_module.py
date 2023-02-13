import os
import torch
import shutil
import pathlib
import requests 

import pandas as pd
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset
from skimage import io
import imageio

from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split


class DatasetModule(torch.utils.data.Dataset): #if False, then test data will be handled
    '''
        Args:
        metadata (str): path to the metadata file
        dataset (str): path to images folder
        train: if True, the train data is handled, otherwise the test data
        transforms:to be used on the data to-do: make it list
        '''
    
    
    def __init__(
        
        self, 
        dataset: str,
        #metadata: str,
        train:str,
        transforms= ToTensor() # data augmentation ^ transformation
    ):

        #self.metadata=pd.read_json(metadata)
        self.transforms=transforms
        self.train=train

        if self.train=='train':
            self.dataset=pd.read_json('metadata/train_df.jsonl',lines=True)

        elif self.train=='val':
            self.dataset=pd.read_json('metadata/val_df.jsonl',lines=True)

        elif self.train=='test':
            self.dataset=pd.read_json('metadata/test_df.jsonl',lines=True)

    def __len__(self):
            return self.dataset.shape[0]
        
    def __getitem__(self, idx: int):
        #metadata=self.metadata.iloc[:,idx].to_dict()
        if self.train=='train' or 'val':
            img_path=f'dataset/train/img/{self.dataset.iloc[idx]["image_id"]}'
            #print('imagepath',img_path)
            image = ToTensor()(imageio.imread(img_path))
            
            msk_path=f'dataset/train/msk/{self.dataset.iloc[idx]["image_id"].replace("IMG","MSK")}'
            mask= ToTensor()(imageio.imread(msk_path))
            return image,mask #,self.metadata[self.dataset.iloc[idx]].to_dict()

        elif self.train=='test':
        
            test_img_path=f'dataset/test/img/{self.dataset.iloc[idx]["image_id"]}'
            #print(test_img_path)
            image = ToTensor()(imageio.imread(test_img_path))
            

            return image#,self.metadata[self.dataset.iloc[idx]].to_dict()
