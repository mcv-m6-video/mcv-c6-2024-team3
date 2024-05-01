""" Dataset class for HMDB51 dataset. """

import os
import random
from enum import Enum
import numpy as np

from glob import glob, escape
import pandas as pd
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2

import cv2
import os
import shutil

class HMDB51Dataset(Dataset):
    """
    Dataset class for HMDB51 dataset.
    """

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


    def __init__(
        self, 
        videos_dir: str,
        modality_name: str, 
        annotations_dir: str, 
        split: Split, 
        regime: Regime, 
        clip_length: int, 
        crop_size: int, 
        temporal_stride: int,
        nt: int,
        ns: int,
        mod_nt: int,
        mod_ns: int,
        mod_clip_length: int,
        mod_crop_size: int,
        mod_temporal_stride: int
    ) -> None:
        """
        Initialize HMDB51 dataset.

        Args:
            videos_dir (str): Directory containing video files.
            annotations_dir (str): Directory containing annotation files.
            split (Split): Dataset split (TEST_ON_SPLIT_1, TEST_ON_SPLIT_2, TEST_ON_SPLIT_3).
            regime (Regimes): Dataset regime (TRAINING, TESTING, VALIDATION).
            split (Splits): Dataset split (TEST_ON_SPLIT_1, TEST_ON_SPLIT_2, TEST_ON_SPLIT_3).
            clip_length (int): Number of frames of the clips.
            crop_size (int): Size of spatial crops (squares).
            temporal_stride (int): Receptive field of the model will be (clip_length * temporal_stride) / FPS.
        """
        self.videos_dir = videos_dir
        self.modality_name = modality_name
        self.annotations_dir = annotations_dir
        self.split = split
        self.regime = regime
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.temporal_stride = temporal_stride
        self.Ns = ns
        self.Nt = nt

        self.mod_clip_length = mod_clip_length
        self.mod_crop_size = mod_crop_size
        self.mod_Nt = mod_nt
        self.mod_Ns = mod_ns
        self.mod_temporal_stride = mod_temporal_stride


        self.annotation = self._read_annotation()
        self.transform = self._create_transform()
        self.transform_mod = self._create_transform_mod()



    def _read_annotation(self) -> pd.DataFrame:
        """
        Read annotation files.

        Returns:
            pd.DataFrame: Dataframe containing video annotations.
        """
        split_suffix = "_test_split" + str(self.split.value) + ".txt"

        annotation = []
        for class_name in HMDB51Dataset.CLASS_NAMES:
            annotation_file = os.path.join(self.annotations_dir, class_name + split_suffix)
            df = pd.read_csv(annotation_file, sep=" ").dropna(axis=1, how='all') # drop empty columns
            df.columns = ['video_name', 'train_or_test']
            df = df[df.train_or_test == self.regime.value]
            df = df.rename(columns={'video_name': 'video_path'})
            df['video_path'] = os.path.join(self.videos_dir, class_name, '') + df['video_path'].replace('\.avi$', '', regex=True)
            df = df.rename(columns={'train_or_test': 'class_id'})
            df['class_id'] = HMDB51Dataset.CLASS_NAMES.index(class_name)
            annotation += [df]

        return pd.concat(annotation, ignore_index=True)


    def _create_transform(self) -> v2.Compose:
        """
        Create transform based on the dataset regime.

        Returns:
            v2.Compose: Transform for the dataset.
        """
        if self.regime == HMDB51Dataset.Regime.TRAINING:
            return v2.Compose([
                v2.Resize(self.crop_size), # Shortest side of the frame to be resized to the given size
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                v2.FiveCrop(size=(self.crop_size,self.crop_size)),
                v2.Lambda(lambda crops: torch.stack([v2.ToDtype(torch.float32, scale=True)(crop) for crop in crops])),  # Convert each crop to tensor
                v2.Lambda(lambda crops: torch.stack([v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops])) 
            ])
        else:
            return v2.Compose([
                #v2.ToDtype(torch.float32, scale=True),
                v2.Resize(self.crop_size), # Shortest side of the frame to be resized to the given size
                v2.FiveCrop(size=(self.crop_size,self.crop_size)),
                v2.Lambda(lambda crops: torch.stack([v2.ToDtype(torch.float32, scale=True)(crop) for crop in crops])),  # Convert each crop to tensor
                v2.Lambda(lambda crops: torch.stack([v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops])) 
            ]) # retorna una cosa de la mida (5, C, H, W)
        
    def _create_transform_mod(self) -> v2.Compose:
        """
        Create transform based on the dataset regime.

        Returns:
            v2.Compose: Transform for the dataset.
        """
        if self.regime == HMDB51Dataset.Regime.TRAINING:
            return v2.Compose([
                v2.Resize(self.mod_crop_size), # Shortest side of the frame to be resized to the given size
                v2.FiveCrop(size=(self.mod_crop_size,self.mod_crop_size)),
                v2.Lambda(lambda crops: torch.stack([v2.ToDtype(torch.float32, scale=True)(crop) for crop in crops])),  # Convert each crop to tensor
            ])
        else:
            return v2.Compose([
                #v2.ToDtype(torch.float32, scale=True),
                v2.Resize(self.mod_crop_size), # Shortest side of the frame to be resized to the given size
                v2.FiveCrop(size=(self.mod_crop_size,self.mod_crop_size)),
                v2.Lambda(lambda crops: torch.stack([v2.ToDtype(torch.float32, scale=True)(crop) for crop in crops])),  # Convert each crop to tensor
            ]) # retorna una cosa de la mida (5, C, H, W)

    def get_num_classes(self) -> int:
        """
        Get the number of classes.

        Returns:
            int: Number of classes.
        """
        return len(HMDB51Dataset.CLASS_NAMES)


    def __len__(self) -> int:
        """
        Get the length (number of videos) of the dataset.

        Returns:
            int: Length (number of videos) of the dataset.
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

        # Get video path from the annotation dataframe and check if it exists
        video_path = df_idx['video_path']
        assert os.path.exists(video_path)

        modality_path = video_path.replace('framesgood', self.modality_name)
        assert os.path.exists(modality_path)

        # Read frames' paths from the video
        frame_paths = sorted(glob(os.path.join(escape(video_path), "*.jpg"))) # get sorted frame paths
        video_len = len(frame_paths)

        modality_paths = sorted(glob(os.path.join(escape(modality_path), "*.npy"))) # get sorted frame paths
        modality_len = len(modality_paths)

        if video_len <= self.clip_length * self.temporal_stride:
            # Not enough frames to create the clip
            clip_begin, clip_end = 0, video_len
        else:
            # Randomly select a clip from the video with the desired length (start and end frames are inclusive)
            clip_begin = random.randint(0, max(video_len - self.clip_length * self.temporal_stride, 0))
            clip_end = clip_begin + self.clip_length * self.temporal_stride

        # Read frames from the video with the desired temporal subsampling
        clips = []
        frames_per_clip = video_len // self.Nt  # Number of frames in each clip

        for k in range(self.Nt):
            clip_start = k * frames_per_clip
            clip_end = clip_start + self.clip_length * self.temporal_stride
            if clip_end >= video_len:
                clip_end = video_len
            video = None
            for i, path in enumerate(frame_paths[clip_start:clip_end:self.temporal_stride]):
                frame = read_image(path)
                if video is None:
                    video = torch.zeros((self.clip_length, 3, frame.shape[1], frame.shape[2]), dtype=torch.uint8)
                video[i] = frame
            clips.append(video)

        # --------------------------------------------------------
        # Lo mismo de arriba pero otra cosa
        if modality_len <= self.mod_clip_length * self.mod_temporal_stride:
            # Not enough frames to create the clip
            clip_begin, clip_end = 0, modality_len
        else:
            # Randomly select a clip from the video with the desired length (start and end frames are inclusive)
            # clip_begin = random.randint(0, max(modality_len - self.clip_length * self.temporal_stride, 0))
            clip_begin = clip_begin
            # clip_end = clip_begin + self.clip_length * self.temporal_stride
            clip_end = clip_end

        clips_modality = []
        frames_per_clip = modality_len // self.mod_Nt  # Number of frames in each clip

        for k in range(self.mod_Nt):
            clip_start = k * frames_per_clip
            clip_end = clip_start + self.mod_clip_length * self.mod_temporal_stride
            if clip_end >= modality_len:
                clip_end = modality_len
            video = None
            for i, path in enumerate(modality_paths[clip_start:clip_end:self.mod_temporal_stride]):
                gm_flow = np.load(path)
                x_gm = gm_flow[:,:,0]
                y_gm = gm_flow[:,:,1]
                zeroes = np.zeros_like(x_gm)
                padded = np.stack((x_gm, y_gm, zeroes), axis = 2)
            
                frame = torch.from_numpy(np.transpose(padded, (2, 0, 1)))  # (C, H, W)
                if video is None:
                    video = torch.zeros((self.mod_clip_length, 3, frame.shape[1], frame.shape[2]), dtype=torch.uint8)
                video[i] = frame
            clips_modality.append(video)
        #-----------------------------------------------------------

        # Get label from the annotation dataframe and make sure video was read
        label = df_idx['class_id']

        return clips, clips_modality, label, video_path

    
    def collate_fn(self, batch: list) -> dict:
        """
        Collate function for creating batches.

        Args:
            batch (list): List of samples.

        Returns:
            dict: Dictionary containing batched clips, labels, and paths.
        """
        # [(clip1, label1, path1), (clip2, label2, path2), ...] 
        #   -> ([clip1, clip2, ...], [label1, label2, ...], [path1, path2, ...])
        unbatched_clips, unbatched_video_modalities, unbatched_labels, paths = zip(*batch)
        # print(unbatched_clips.shape)
        # Apply transformation and permute dimensions: (T, C, H, W) -> (C, T, H, W)
        transformed_clips = [] # llista de clips general
        transformed_video_modalities = []

        for unbatched_clip in unbatched_clips: #unbatched clips te tots els clips d1 video
            
            
            #print("wuantitat de clips que marriben", len(unbatched_clip))
            #print(len(unbatched_clip), "llargda unbatched clip")
            all_clips_one_video = []
            for clip in unbatched_clip: # un clip es un VIDEO
                transformed_frames = self.transform(clip) #AIXO EN TEORIA ES EL VIDEO JA TRANSFORMAT, DE LLARGADA N_FRAMES
                transformed_frames = list(transformed_frames)[::-1] # AIXO SON ELS 5 VIDEOS

                for i in range(0, self.Ns):
                    all_clips_one_video.append(transformed_frames[i].permute(1, 0, 2, 3))
            #print("all clips one video", len(all_clips_one_video)) #aixo ha de tenir mida ns*nt
            transformed_clips.append(all_clips_one_video)

        for unbatched_video_modality in unbatched_video_modalities:
            all_clips_one_video = []
            for clip in unbatched_video_modality:
                transformed_frames = self.transform_mod(clip)
                transformed_frames = list(transformed_frames)[::-1]

                for i in range(0, self.mod_Ns):
                    all_clips_one_video.append(transformed_frames[i].permute(1, 0, 2, 3))
                
            transformed_video_modalities.append(all_clips_one_video)

        # print(np.array(transformed_clips).shape)
        # Concatenate clips along the batch dimension: 
        # B * [(C, T, H, W)] -> B * [(1, C, T, H, W)] -> (B, C, T, H, W)

        batched_clips = []
        for transformed_clip in transformed_clips:
            batched_clip = torch.cat([d.unsqueeze(0) for d in transformed_clip], dim=0)
            batched_clips.append(batched_clip)

        batched_video_modalities = []
        for transformed_video_modality in transformed_video_modalities:
            batched_video_modality = torch.cat([d.unsqueeze(0) for d in transformed_video_modality], dim=0)
            batched_video_modalities.append(batched_video_modality)

        return dict(
            clips=torch.stack(batched_clips, dim=0), # (B, C, T, H, W)
            video_modalities=torch.stack(batched_video_modalities, dim=0),
            labels=torch.tensor(unbatched_labels), # (K,)
            paths=paths  # no need to make it a tensor
        )

def reset_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

if __name__ == '__main__':
    dataset = HMDB51Dataset(
        videos_dir='/ghome/group03/C6/framesgood',
        modality_name='frames',
        annotations_dir='/ghome/group03/C6/MCV-M6-ActionClassificationTask/data/hmdb51/testTrainMulti_601030_splits',
        split=HMDB51Dataset.Split.TEST_ON_SPLIT_1,
        regime=HMDB51Dataset.Regime.TESTING,
        clip_length=16,
        crop_size=112,
        temporal_stride=2,
        nt=5,
        ns=3
    )
    
    dataloader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,  # Shuffle only for training dataset
            collate_fn=dataset.collate_fn,
            num_workers=1,
            pin_memory=True
        )
    
    for i in dataloader:
        clips = i['clips']
        video_modalites = i['video_modalities']
        labels = i['labels']

        print(labels)
        print(labels.shape)

        reset_folder('clips')
        print(clips.shape)
        tensor = clips
        for batch_idx in range(tensor.size(0)):
            for clips_idx in range(tensor.size(1)):
                for clip_idx in range(tensor.size(2)):
                    # Select the image
                    image = tensor[batch_idx, clips_idx, :, clip_idx]
                    # Convert to numpy
                    image = image.numpy()
                    # Transpose the image
                    image = np.transpose(image, (1, 2, 0))
                    # Convert to OpenCV BGR format
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    # Display the image
                    cv2.imwrite(f'clips/clip{batch_idx}_{clips_idx}_{clip_idx}.jpg', image * 255)

        reset_folder('clips_modality')
        tensor = video_modalites
        for batch_idx in range(tensor.size(0)):
            for clips_idx in range(tensor.size(1)):
                for clip_idx in range(tensor.size(2)):
                    # Select the image
                    image = tensor[batch_idx, clips_idx, :, clip_idx]
                    # Convert to numpy
                    image = image.numpy()
                    # Transpose the image
                    image = np.transpose(image, (1, 2, 0))
                    # Convert to OpenCV BGR format
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    # Display the image
                    cv2.imwrite(f'clips_modality/clip{batch_idx}_{clips_idx}_{clip_idx}.jpg', image * 255)
        
        break
        