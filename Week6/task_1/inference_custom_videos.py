import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Iterator

from torch.utils.data import DataLoader

from datasets.HMDB51Dataset import HMDB51Dataset
from models import model_creator
from utils import model_analysis
from utils import statistics
import numpy as np

from pytorchtools import EarlyStopping

import matplotlib.pyplot as plt
from torchvision.io import ImageReadMode
from torchvision.io import read_image

from torchvision.transforms import v2

import os
import pickle
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
import torch.nn.utils.prune as prune
from movinets import MoViNet
from movinets.config import _C

import wandb
import random
import cv2
import shutil

import random


from glob import glob

small_model = {
    'clip_length': 4,
    'crop_size' : 182,
    'temporal_stride': 12
}

big_model = {
    "crop_size": 256,
    "clip_length": 16,
    "temporal_stride": 5
}

def create_datasets(
        frames_dir: str,
        annotations_dir: str,
        split: HMDB51Dataset.Split,
        clip_length: int,
        crop_size: int,
        temporal_stride: int
) -> Dict[str, HMDB51Dataset]:
    """
    Creates datasets for training, validation, and testing.

    Args:
        frames_dir (str): Directory containing the video frames (a separate directory per video).
        annotations_dir (str): Directory containing annotation files.
        split (HMDB51Dataset.Split): Dataset split (TEST_ON_SPLIT_1, TEST_ON_SPLIT_2, TEST_ON_SPLIT_3).
        clip_length (int): Number of frames of the clips.
        crop_size (int): Size of spatial crops (squares).
        temporal_stride (int): Receptive field of the model will be (clip_length * temporal_stride) / FPS.

    Returns:
        Dict[str, HMDB51Dataset]: A dictionary containing the datasets for training, validation, and testing.
    """
    datasets = {}
    for regime in HMDB51Dataset.Regime:
        datasets[regime.name.lower()] = HMDB51Dataset(
            frames_dir,
            annotations_dir,
            split,
            regime,
            clip_length,
            crop_size,
            temporal_stride,
        )
    
    return datasets

def create_dataset(video_path, clip_length, temporal_stride, crop_size):
    transform = v2.Compose([
                v2.Resize(crop_size), # Shortest side of the frame to be resized to the given size
                v2.CenterCrop(crop_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    save_frames = 'temporal'
    # If folder exists, reset
    if os.path.exists(save_frames):
        shutil.rmtree(save_frames)

    os.makedirs(save_frames)

    video = cv2.VideoCapture(video_path)
    counter = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        cv2.imwrite(f'{save_frames}/{counter}.jpg', frame)
        counter += 1

    frame_paths = sorted(glob(os.path.join(save_frames, "*.jpg"))) # get sorted frame paths
    video_len = len(frame_paths)

    if video_len <= clip_length * temporal_stride:
        # Not enough frames to create the clip
        clip_begin, clip_end = 0, video_len
    else:
        # Randomly select a clip from the video with the desired length (start and end frames are inclusive)
        clip_begin = random.randint(0, max(video_len - clip_length * temporal_stride, 0))
        clip_end = clip_begin + clip_length * temporal_stride

    # Read frames from the video with the desired temporal subsampling
    video = None
    for i, path in enumerate(frame_paths[clip_begin:clip_end:temporal_stride]):
        frame = read_image(path, ImageReadMode.RGB)  # (C, H, W)
        if video is None:
            video = torch.zeros((clip_length, 3, frame.shape[1], frame.shape[2]), dtype=torch.uint8)
        video[i] = frame

    # Get label from the annotation dataframe and make sure video was read
    assert video is not None

    clip = video
    # Apply transformation and permute dimensions: (T, C, H, W) -> (C, T, H, W)
    transformed_clips = transform(clip).permute(1, 0, 2, 3)
    # print(np.array(transformed_clips).shape)
    # Concatenate clips along the batch dimension: 
    # B * [(C, T, H, W)] -> B * [(1, C, T, H, W)] -> (B, C, T, H, W)
    batched_clips = transformed_clips.unsqueeze(0)

    return batched_clips

def create_model(type_model):
    num_classes = len(HMDB51Dataset.CLASS_NAMES)
    load_pretrain = True

    if type_model == 'small_model':
        model = MoViNet(_C.MODEL.MoViNetA0, causal = True, pretrained = load_pretrain)

        # print(model.blocks[-1])
        for i in range(1,5):
            # print(i)
            model.blocks[-i] = torch.nn.Identity()
        # print(model.blocks[-1])
        # for name, module in model.blocks[3].named_modules():
        #     print(module)
        #     module = torch.nn.Identity()
        #     print(module)
        # print(model)
        # model = modify_x3d_model(model)
        model.conv7.conv_1.conv2d = torch.nn.Conv2d(56, 480, kernel_size=(1, 1), stride=(1, 1), bias = False)
        model.classifier[0].conv_1.conv2d = torch.nn.Conv2d(480, 64, kernel_size=(1, 1), stride=(1, 1))
        model.classifier[3] = torch.nn.Conv3d(64, num_classes, (1,1,1))

        model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)

    if type_model == 'big_model':
        model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=load_pretrain)
        model.blocks[5].proj = nn.Identity()
        model =  nn.Sequential(
            model,
            nn.Linear(2048, num_classes, bias=True),
        )
    
    return model

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    model_type = 'big_model' # 'small_model' or 'big_model'

    if model_type == 'small_model':
        config = small_model
    elif model_type == 'big_model':
        config = big_model

    model = create_model(model_type)

    weights = f'model_weights/{model_type}.pth'

    model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
    # model.to(device)
    model.eval()

    folder_path = "/ghome/group03/C6/task2/custom_videos"
    files = os.listdir(folder_path)
    mp4_files = [file for file in files if file.endswith(".mp4")]
    video_paths = sorted([os.path.join(folder_path, file) for file in mp4_files])

    for video_path in video_paths:
        dataset_video = create_dataset(video_path, config['clip_length'], config['temporal_stride'], config['crop_size'])
        if model_type == 'small_model':
            model.clean_activation_buffers()

        with torch.no_grad():
            outputs = model(dataset_video)

            if model_type == 'small_model':
                model.clean_activation_buffers()
            
            predicted = outputs.argmax(dim=1).cpu().numpy()

            # Convert outputs to probablities
            probs = torch.nn.functional.softmax(outputs, dim=1)

            # Get the top-5 largest probabilities
            top5_probs, top5_classes = torch.topk(probs, 5)

            results = [(HMDB51Dataset.CLASS_NAMES[top5_classes[0][i]], top5_probs[0][i].item()) for i in range(5)]

            print(f'Video: {video_path}\nPredicted: {HMDB51Dataset.CLASS_NAMES[predicted[0]]}\nTop-5:')

            for label, prob in results:
                print(f'{label}: {prob:.2f}')




    
    
