""" Dataset class for HMDB51 dataset. """

import os
import random
from enum import Enum
import numpy as np

from glob import glob, escape
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2


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
        annotations_dir: str, 
        split: Split, 
        regime: Regime, 
        clip_length: int, 
        crop_size: int, 
        temporal_stride: int,
        num_seg: int
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
        self.annotations_dir = annotations_dir
        self.split = split
        self.regime = regime
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.temporal_stride = temporal_stride
        self.num_seg = num_seg

        self.annotation = self._read_annotation()
        self.transform = self._create_transform()


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
            
            # transform_list = [
            #     v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            #     v2.RandomGrayscale(),
            #     v2.RandomInvert(),
            #     v2.RandomPosterize(bits=4),
            #     v2.RandomEqualize(),
            # ]
            
            return v2.Compose([
                v2.RandomResizedCrop(self.crop_size),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return v2.Compose([
                v2.Resize(self.crop_size), # Shortest side of the frame to be resized to the given size
                v2.CenterCrop(self.crop_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


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

        # Read frames' paths from the video
        frame_paths = sorted(glob(os.path.join(escape(video_path), "*.jpg"))) # get sorted frame paths
        # video_len = len(frame_paths)

        video_segments = self.split_list(frame_paths, self.num_seg)
        
        max_len_seg = len(max(video_segments, key=lambda x:len(x)))

        videos = []
        labels = []
        video_paths = []

        for video_segment in video_segments:
            # print(np.array(video_segment).shape)
            segment_len = len(video_segment)

            if segment_len < max_len_seg:
                video_segment.append(video_segment[-1])
                segment_len = len(video_segment)

            # print('LEN VIDEO SEGMENT AFTER: ', len(video_segment))
            if segment_len <= self.clip_length * self.temporal_stride:
                # Not enough frames to create the clip
                clip_begin, clip_end = 0, segment_len
            else:
                # Randomly select a clip from the video with the desired length (start and end frames are inclusive)
                clip_begin = random.randint(0, max(segment_len - self.clip_length * self.temporal_stride, 0))
                clip_end = clip_begin + self.clip_length * self.temporal_stride

            # Read frames from the video with the desired temporal subsampling
            video = None
            for i, path in enumerate(video_segment[clip_begin:clip_end:self.temporal_stride]):
                frame = read_image(path)  # (C, H, W)
                if video is None:
                    video = torch.zeros((self.clip_length, 3, frame.shape[1], frame.shape[2]), dtype=torch.uint8)
                video[i] = frame

            # Get label from the annotation dataframe and make sure video was read
            label = df_idx['class_id']
            assert video is not None

            videos.append(video)

        videos = torch.cat(videos)

        return videos, label, video_path

    def split_list(self, lst, N):
        k, m = divmod(len(lst), N)
        return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(N)]

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

        unbatched_clips, unbatched_labels, paths = zip(*batch)
        # print('Unbatched clips shape: ' , unbatched_clips[0].shape)

        segmented_clips = [torch.tensor_split(clip, self.num_seg, dim=0) for clip in unbatched_clips]
        # print('Segmented clips shape: ' , segmented_clips[0][0].shape)

        # Apply transformation and permute dimensions: (T, C, H, W) -> (C, T, H, W)
        transformed_clips = [torch.cat([self.transform(clip).permute(1, 0, 2, 3).unsqueeze(0) for clip in segment], dim = 0) for segment in segmented_clips]
        
        # [(3, C, T, H, W)]
        # Concatenate clips along the batch dimension: 
        # B * [(C, T, H, W)] -> B * [(1, C, T, H, W)] -> (B, C, T, H, W)
        # FINAL -> (B, 3, C, T, H, W)
        batched_clips = torch.cat([d.unsqueeze(0) for d in transformed_clips], dim=0)
        
        return dict(
            clips=batched_clips, # (B, C, T, H, W)
            labels=torch.tensor(unbatched_labels), # (K,)
            paths=paths  # no need to make it a tensor
        )


class HMDB51Dataset_baseline(Dataset):
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
        annotations_dir: str, 
        split: Split, 
        regime: Regime, 
        clip_length: int, 
        crop_size: int, 
        temporal_stride: int
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
        self.annotations_dir = annotations_dir
        self.split = split
        self.regime = regime
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.temporal_stride = temporal_stride

        self.annotation = self._read_annotation()
        self.transform = self._create_transform()


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
                v2.RandomResizedCrop(self.crop_size),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return v2.Compose([
                v2.Resize(self.crop_size), # Shortest side of the frame to be resized to the given size
                v2.CenterCrop(self.crop_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


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

        # Read frames' paths from the video
        frame_paths = sorted(glob(os.path.join(escape(video_path), "*.jpg"))) # get sorted frame paths
        video_len = len(frame_paths)

        if video_len <= self.clip_length * self.temporal_stride:
            # Not enough frames to create the clip
            clip_begin, clip_end = 0, video_len
        else:
            # Randomly select a clip from the video with the desired length (start and end frames are inclusive)
            clip_begin = random.randint(0, max(video_len - self.clip_length * self.temporal_stride, 0))
            clip_end = clip_begin + self.clip_length * self.temporal_stride

        # Read frames from the video with the desired temporal subsampling
        video = None
        for i, path in enumerate(frame_paths[clip_begin:clip_end:self.temporal_stride]):
            frame = read_image(path)  # (C, H, W)
            if video is None:
                video = torch.zeros((self.clip_length, 3, frame.shape[1], frame.shape[2]), dtype=torch.uint8)
            video[i] = frame

        # Get label from the annotation dataframe and make sure video was read
        label = df_idx['class_id']
        assert video is not None

        return video, label, video_path

    
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
        unbatched_clips, unbatched_labels, paths = zip(*batch)
        # print(unbatched_clips.shape)
        # Apply transformation and permute dimensions: (T, C, H, W) -> (C, T, H, W)
        transformed_clips = [self.transform(clip).permute(1, 0, 2, 3) for clip in unbatched_clips]
        for sub in transformed_clips:
            print(type(sub))
            for s in sub:
                print(type(s))
                for a in s:
                    print(type(a))
        # print(np.array(transformed_clips).shape)
        # Concatenate clips along the batch dimension: 
        # B * [(C, T, H, W)] -> B * [(1, C, T, H, W)] -> (B, C, T, H, W)
        batched_clips = torch.cat([d.unsqueeze(0) for d in transformed_clips], dim=0)
        print(batched_clips.shape)

        return dict(
            clips=batched_clips, # (B, C, T, H, W)
            labels=torch.tensor(unbatched_labels), # (K,)
            paths=paths  # no need to make it a tensor
        )
