import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import os
from enum import Enum
from torchvision.io import ImageReadMode
from torchvision.io import read_image
from torch.utils.data import DataLoader
import numpy as np

import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt


class MyDataset(Dataset):
    CLASS_NAMES = [
        "brush_hair", "catch", "clap", "climb_stairs", "draw_sword", "drink", 
        "fall_floor", "flic_flac", "handstand", "hug", "kick", "kiss", "pick", 
        "pullup", "push", "ride_bike", "run", "shoot_ball", "shoot_gun", "situp", 
        "smoke", "stand", "sword", "talk", "turn", "wave", 
        "cartwheel", "chew", "climb", "dive", "dribble", "eat", "fencing", 
        "golf", "hit", "jump", "kick_ball", "laugh", "pour", "punch", "pushup", 
        "ride_horse", "shake_hands", "shoot_bow", "sit", "smile", "somersault", 
        "swing_baseball", "sword_exercise", "throw", "walk"
    ]
    
    class Split(Enum):
        """
        Enum class for dataset splits.
        """
        TEST_ON_SPLIT_1 = 1
        TEST_ON_SPLIT_2 = 2
        TEST_ON_SPLIT_3 = 3
        
    class Regime(Enum):
        """
        Enum class for dataset regimes.
        """
        TRAINING = 1
        TESTING = 2
        VALIDATION = 3
    
    def __init__(self, annotations_dir, videos_dir, split, regime, transform = None):
        self.annotations_dir = annotations_dir
        self.videos_dir = videos_dir
        self.split = split
        self.regime = regime
        
        self.annotation = self._read_annotation()
        self.transform = transform
        
    def _read_annotation(self) -> pd.DataFrame:
        """
        Read annotation files.

        Returns:
            pd.DataFrame: Dataframe containing video annotations.
        """
        split_suffix = "_test_split" + str(self.split.value) + ".txt"

        annotation = []
        for class_name in MyDataset.CLASS_NAMES:
            annotation_file = os.path.join(self.annotations_dir, class_name + split_suffix)
            df = pd.read_csv(annotation_file, sep=" ").dropna(axis=1, how='all') # drop empty columns
            df.columns = ['video_name', 'train_or_test']
            df = df[df.train_or_test == self.regime.value]
            df = df.rename(columns={'video_name': 'video_folder'})  # Rename to 'video_folder'
            df['video_folder'] = os.path.join(self.videos_dir, class_name, '') + df['video_folder'].replace('\.avi$', '', regex=True)
            df = df.rename(columns={'train_or_test': 'class_id'})
            df['class_id'] = MyDataset.CLASS_NAMES.index(class_name)

            # Iterate over each row to get the full paths of images within each video folder
            image_paths = []
            for _, row in df.iterrows():
                video_folder = row['video_folder']
                # Check if the folder exists
                if os.path.exists(video_folder):
                    # Get all files within the folder (assuming they are images)
                    files = sorted(os.listdir(video_folder))
                    # Filter only image files
                    image_files = [os.path.join(video_folder, f) for f in files if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
                    image_paths.extend(image_files)
            # Create a new DataFrame with full image paths
            image_df = pd.DataFrame({'image_path': image_paths, 'class_id': df['class_id'].iloc[0]})  # Assuming all images in a folder belong to the same class
            annotation.append(image_df)

        return pd.concat(annotation, ignore_index=True)
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.annotation)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get item (video) from the dataset.

        Args:
            idx (int): Index of the item (video).

        Returns:
            tuple: Tuple containing video, label, and video path.
        """
        df_idx = self.annotation.iloc[idx]
        
        img_filename = df_idx['image_path']
        label = df_idx['class_id']
        
        image = read_image(img_filename, ImageReadMode.RGB) 
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
    
class MyDatasetInference(Dataset):
    CLASS_NAMES = [
        "brush_hair", "catch", "clap", "climb_stairs", "draw_sword", "drink", 
        "fall_floor", "flic_flac", "handstand", "hug", "kick", "kiss", "pick", 
        "pullup", "push", "ride_bike", "run", "shoot_ball", "shoot_gun", "situp", 
        "smoke", "stand", "sword", "talk", "turn", "wave", 
        "cartwheel", "chew", "climb", "dive", "dribble", "eat", "fencing", 
        "golf", "hit", "jump", "kick_ball", "laugh", "pour", "punch", "pushup", 
        "ride_horse", "shake_hands", "shoot_bow", "sit", "smile", "somersault", 
        "swing_baseball", "sword_exercise", "throw", "walk"
    ]
    
    class Split(Enum):
        """
        Enum class for dataset splits.
        """
        TEST_ON_SPLIT_1 = 1
        TEST_ON_SPLIT_2 = 2
        TEST_ON_SPLIT_3 = 3
        
    class Regime(Enum):
        """
        Enum class for dataset regimes.
        """
        TRAINING = 1
        TESTING = 2
        VALIDATION = 3
    
    def __init__(self, annotations_dir, videos_dir, split, regime, transform = None):
        self.annotations_dir = annotations_dir
        self.videos_dir = videos_dir
        self.split = split
        self.regime = regime
        
        self.annotation = self._read_annotation()
        self.transform = transform
        
    def _read_annotation(self) -> pd.DataFrame:
        """
        Read annotation files.

        Returns:
            pd.DataFrame: DataFrame containing video annotations.
        """
        split_suffix = "_test_split" + str(self.split.value) + ".txt"

        annotation = []
        for class_name in MyDataset.CLASS_NAMES:
            annotation_file = os.path.join(self.annotations_dir, class_name + split_suffix)
            df = pd.read_csv(annotation_file, sep=" ").dropna(axis=1, how='all')  # drop empty columns
            df.columns = ['video_name', 'train_or_test']
            df = df[df.train_or_test == self.regime.value]
            df = df.rename(columns={'video_name': 'video_folder'})  # Rename to 'video_folder'
            df['video_folder'] = os.path.join(self.videos_dir, class_name, '') + df['video_folder'].replace('\.avi$', '', regex=True)
            df = df.rename(columns={'train_or_test': 'class_id'})
            df['class_id'] = MyDataset.CLASS_NAMES.index(class_name)

            # Iterate over each row to get the full paths of images within each video folder
            video_paths = []
            for _, row in df.iterrows():
                video_folder = row['video_folder']
                # Check if the folder exists
                if os.path.exists(video_folder):
                    # Get all files within the folder (assuming they are images)
                    files = sorted(os.listdir(video_folder))
                    # Filter only image files
                    image_files = [os.path.join(video_folder, f) for f in files if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
                    # Concatenate all image paths into a single string
                    video_path = ','.join(image_files)
                    video_paths.append(video_path)
            # Create a new DataFrame with video paths
            video_df = pd.DataFrame({'video_path': video_paths, 'class_id': df['class_id'].iloc[0]})  # Assuming all images in a folder belong to the same class
            annotation.append(video_df)

        return pd.concat(annotation, ignore_index=True)
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.annotation)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get item (video) from the dataset.

        Args:
            idx (int): Index of the item (video).

        Returns:
            tuple: Tuple containing video, label, and video path.
        """
        df_idx = self.annotation.iloc[idx]
        
        img_filenames = df_idx['video_path']
        label = df_idx['class_id']
        
        all_images = []
        
        for img_filename in img_filenames.split(','):
            image = read_image(img_filename, ImageReadMode.RGB)
            if self.transform is not None:
                image = self.transform(image)
            all_images.append(image)
        
        print(len(all_images))
        return all_images, label

        
        
    
if __name__ == '__main__':
    annotations_dir = "data/hmdb51/testTrainMulti_601030_splits"
    frames_dir = '../framesgood'
    split = MyDataset.Split.TEST_ON_SPLIT_1
    regime = MyDataset.Regime.VALIDATION
    mydataset = MyDatasetInference(annotations_dir, frames_dir, split, regime)

    tabla = mydataset._read_annotation()

    tabla.to_csv('tabla.csv')

    imgs, label = mydataset[0]


    print(len(mydataset))

    '''
    dataloader = DataLoader(mydataset, batch_size=3, shuffle=False)

    for i, (imgs, label) in enumerate(dataloader):
        print(i)
        print(imgs)
        break

    print(type(imgs))
    print(label)
    '''
    
    