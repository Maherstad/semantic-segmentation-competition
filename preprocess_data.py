import os
import shutil
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split


# Create the main directory if it doesn't exist


def preprocess(source:str='./raw_dataset', destination:str='./dataset'):

    # Create a dataframe with image_id, zone, and domain 
    # and move all images from subdirectories to the main directory 

    pathlib.Path('./dataset/train/img/').mkdir(parents=True, exist_ok=True) 
    pathlib.Path('./dataset/train/msk/').mkdir(parents=True, exist_ok=True) 
    pathlib.Path('./dataset/test/img/').mkdir(parents=True, exist_ok=True) 
    pathlib.Path('./metadata/').mkdir(parents=True, exist_ok=True) 




    dataframe_construction_list = []
    dataframe_test_img_construction_list=[]
    # Split the source directory into train and test
    
    for file in ['train','test']:
        if file == 'train':
            domains = [i for i in os.listdir(f'{source}/{file}') if not i.startswith('.')]
            for domain in domains:
                zones = [i for i in os.listdir(f'{source}/{file}/{domain}') if not i.startswith('.')]
                for zone in zones:
                    # Iterate over the image files in the img directory
                    img_dir = [i for i in os.listdir(f'{source}/{file}/{domain}/{zone}/img') if not i.startswith('.') 
                               and i.endswith('.tif')
                              ]
                    for img_id in img_dir:
                        dataframe_construction_list.append({'image_id': img_id,
                                                            'domain_zone': domain + '_' +zone
                                                           })
                        file_name = f'{source}/{file}/{domain}/{zone}/img/{img_id}'
                        shutil.copy(file_name, './dataset/train/img')

                    # Iterate over the mask files in the msk directory
                    msk_dir = [i for i in os.listdir(f'{source}/{file}/{domain}/{zone}/msk') if not i.startswith('.')
                               and i.endswith('.tif')
                              ]
                    for msk_id in msk_dir:
                        file_name = f'{source}/{file}/{domain}/{zone}/msk/{msk_id}'
                        shutil.copy(file_name, './dataset/train/msk')
        elif file == 'test':
            # Move all test images to ./test/img
            domains = [i for i in os.listdir(f'{source}/{file}') if not i.startswith('.')]
            for domain in domains:
                zones = [i for i in os.listdir(f'{source}/{file}/{domain}') if not i.startswith('.')]
                for zone in zones:
                    ground_dir=[i for i in os.listdir(f'{source}/{file}/{domain}/{zone}/img') if not i.startswith('.')
                                and  i.endswith('.tif')
                               ]
                    for tif_file in ground_dir:
                        dataframe_test_img_construction_list.append({'image_id': tif_file,
                                                            'domain_zone': domain + '_' +zone
                                                           })
                        if len(dataframe_test_img_construction_list)%1000==0:
                            print(len(dataframe_test_img_construction_list))
                        file_name = f'{source}/{file}/{domain}/{zone}/img/{tif_file}'
                        shutil.copy(file_name, './dataset/test/img')


    img_ids=pd.DataFrame(dataframe_construction_list,columns=['image_id','domain_zone'])  
    img_ids.to_json('./metadata/img_domain_zone_combination.jsonl',orient='records',lines=True) 
    
    test_ids=pd.DataFrame(dataframe_test_img_construction_list,columns=['image_id','domain_zone'])  
    test_ids.to_json('./metadata/test_df.jsonl',orient='records',lines=True)

    #  image_id domain	zone
    #0	012527	D016_2020	Z2_AA
    #1	012777	D016_2020	Z4_UU
    #2	013958	D016_2020	Z15_FN


def train_dev_split(dataframe_path:str='./metadata/img_domain_zone_combination.jsonl',toy_dataset=False):
    #input is the name of the dataframe containing the combinations of img and domain-zone
    #output is a 2 Dataframes: one for the training data and one for the validation data
    df=pd.read_json(dataframe_path,lines=True)
    
    X=df['image_id']
    y=df['domain_zone']
    if toy_dataset:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0,stratify=y)
    
    #save X_train,y_train to a dataframe
    train_df = pd.DataFrame({'image_id': X_train, 'domain_zone': y_train})
    train_df.to_json('./metadata/train_df.jsonl', orient='records',lines=True)

    
    #save X_val,y_val to another dataframe
    val_df = pd.DataFrame({'image_id': X_val, 'domain_zone': y_val})
    val_df.to_json('./metadata/val_df.jsonl', orient='records',lines=True)


def delete_non_tif_images(dir_list:list=['./dataset/train/img','./dataset/train/msk','./dataset/test/img']):
    for directory in dir_list:
        for filename in os.listdir(directory):
            if not filename.endswith(".tif"):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    print('function delete_non_tif_images executed')

if __name__ == "__main__":
    preprocess()
    train_dev_split(toy_dataset=True)
    delete_non_tif_images()
    