import os
import torch
import shutil
import pathlib
import requests 
import rasterio
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

from torch.utils.data import Dataset

from skimage import io
from skimage import img_as_float

import imageio

import torchvision.transforms as tt 
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
        transforms= None, # data augmentation ^ transformation
        num_classes=13
    ):

        #self.metadata=pd.read_json(metadata)
        self.transforms= None #tt.Compose([tt.ToTensor()]) #tt.Resize(224), 
        self.train=train
        self.num_classes=num_classes
        if self.train=='train':
            self.dataset=pd.read_json('metadata/train_df.jsonl',lines=True)

        elif self.train=='val':
            self.dataset=pd.read_json('metadata/val_df.jsonl',lines=True)

        elif self.train=='test':
            self.dataset=pd.read_json('metadata/test_df.jsonl',lines=True)

            
    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read()
            return array

    def read_msk(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_msk:
            array = src_msk.read()[0]
            array[array > self.num_classes] = self.num_classes
            array = array-1
            array = np.stack([array == i for i in range(self.num_classes)], axis=0)
            return array
        
        
    def __len__(self):
            return self.dataset.shape[0]
        
    def __getitem__(self, idx: int):
        #metadata=self.metadata.iloc[:,idx].to_dict()
        if self.train=='train' or 'val':
            img_path=f'dataset/train/img/{self.dataset.iloc[idx]["image_id"]}'
            #print('imagepath',img_path)
            #img = imageio.imread(img_path)
            #image = self.transforms(img)
            image=self.read_img(img_path)
            image = img_as_float(image)
            #image=self.transforms(image)#.reshape(512,512,5)
            image=torch.as_tensor(image, dtype=torch.float32)
            image=image.unsqueeze(0)
            image = torch.nn.functional.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)
            image=image.squeeze(0)

            
            
            
            msk_path=f'dataset/train/msk/{self.dataset.iloc[idx]["image_id"].replace("IMG","MSK")}'
            #mask = imageio.imread(msk_path)
            #mask= self.transforms(mask)
            
            mask=self.read_msk(msk_path)
            mask = img_as_float(mask)
            
            mask=torch.as_tensor(mask, dtype=torch.float32)
            mask=mask.unsqueeze(0)
            mask = torch.nn.functional.interpolate(mask, size=(256, 256), mode='bilinear', align_corners=False)
            mask=mask.squeeze(0)
            
            
            
            #mask=self.transforms(mask)#.reshape(512,512,13)

            return torch.as_tensor(image, dtype=torch.float),torch.as_tensor(mask, dtype=torch.float) #,self.metadata[self.dataset.iloc[idx]].to_dict()

        elif self.train=='test':
        
            test_img_path=f'dataset/test/img/{self.dataset.iloc[idx]["image_id"]}'
            #print(test_img_path)
            #img = imageio.imread(test_img_path)
            #image = self.transforms(img)
            image=self.read_image(test_img_path)
            image = img_as_float(image)
            #image=self.transforms(image).reshape(512,512,5)
            image=torch.as_tensor(image, dtype=torch.float32)
            image=image.unsqueeze(0)
            image = torch.nn.functional.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)
            image=image.squeeze(0)

            return torch.as_tensor(image, dtype=torch.float)#,self.metadata[self.dataset.iloc[idx]].to_dict()
