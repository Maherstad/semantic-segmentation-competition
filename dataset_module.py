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
import albumentations as A
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
        num_classes=13,
        use_augmentations=True
    ):

        #self.metadata=pd.read_json(metadata)
        self.transforms= None #tt.Compose([tt.ToTensor()]) #tt.Resize(224),
        self.train=train
        self.num_classes=num_classes
        self.use_augmentations=use_augmentations

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
            array = src_msk.read()#[0]
            array[array > self.num_classes] = self.num_classes
            array = array-1
            #array = np.stack([array == i for i in range(self.num_classes)], axis=0)
            return array


    def __len__(self):
            return self.dataset.shape[0]

    def __getitem__(self, idx: int):
        #metadata=self.metadata.iloc[:,idx].to_dict()
        if self.train=='train' or 'val':
            img_path=f'dataset/train/img/{self.dataset.iloc[idx]["image_id"]}'
            image=self.read_img(img_path)

            msk_path=f'dataset/train/msk/{self.dataset.iloc[idx]["image_id"].replace("IMG","MSK")}'
            mask=self.read_msk(msk_path)
            #mask = mask.astype(np.uint8) * 255


            transform_set = A.Compose([ A.Resize(width=256,height=256),
                                            A.VerticalFlip(p=0.5),
                                            A.HorizontalFlip(p=0.5),
                                            A.RandomRotate90(p=0.5),

                                      ]  #A.Normalize(mean=[0,0,0,0,0],std=[1,1,1,1,1],max_pixel_value=255,p=1.0)
                                             )


            if self.use_augmentations:
                sample = {"image" : image.swapaxes(0, 2).swapaxes(0, 1), "mask": mask.swapaxes(0, 2).swapaxes(0, 1)}
                transformed_sample = transform_set(**sample)
                image, mask = transformed_sample["image"].swapaxes(0, 2).swapaxes(1, 2).copy(), transformed_sample["mask"].swapaxes(0, 2).swapaxes(1, 2).copy()


            image = img_as_float(image)
            #mask = mask.astype(bool)

            image=torch.as_tensor(image, dtype=torch.float)
            mask=torch.as_tensor(mask, dtype=torch.long)

            #print(f'XXXXXXXXXX  {type(image)} image containing {image.dtype} and shape is {image.shape}')
            #print(f'XXXXXXXXXX  {type(mask)} mask containing {mask.dtype} and shape is {mask.shape}')
            return image,mask   #,self.metadata[self.dataset.iloc[idx]].to_dict()

        elif self.train=='test':

            test_img_path=f'dataset/test/img/{self.dataset.iloc[idx]["image_id"]}'
            image=self.read_image(test_img_path)


            transform_set = A.Compose([ A.Resize(width=256,height=256),
                                            A.VerticalFlip(p=0.5),
                                            A.HorizontalFlip(p=0.5),
                                            A.RandomRotate90(p=0.5),
                                          ]#A.Normalize(mean=[0,0,0,0,0],std=[1,1,1,1,1],max_pixel_value=255,p=1.0)

                                             )
            if self.use_augmentations:
                sample = {"image" : image.swapaxes(0, 2).swapaxes(0, 1)}
                transformed_sample = transform_set(**sample)
                image = transformed_sample["image"].swapaxes(0, 2).swapaxes(1, 2).copy()


            image = img_as_float(image)

            image=torch.as_tensor(image, dtype=torch.float)

            return image#,self.metadata[self.dataset.iloc[idx]].to_dict()
