""" Main script for training a video classification model on HMDB51 dataset. 

Early stopping code from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb

"""

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Iterator

from torch.utils.data import DataLoader

from datasets.HMDB51DatasetCustom import HMDB51Dataset31, HMDB51Dataset32, HMDB51Dataset
from models import model_creator
from utils import model_analysis
from utils import statistics
import numpy as np

from pytorchtools import EarlyStopping

import matplotlib.pyplot as plt

import os
import pickle

import wandb

import cv2

configuration = {}

def plot_label_accuracies(predictions, labels, class_names, sufix, root):
    """
    Plot the accuracy of each label.

    Args:
    - predictions (numpy.ndarray): Array of predicted labels.
    - labels (numpy.ndarray): Array of true labels.
    - class_names (list): List of class names corresponding to the labels.

    Returns:
    - None
    """
    # Calculate accuracy for each label
    num_labels = np.max(labels) + 1
    accuracies = []
    for label in range(num_labels):
        correct_predictions = np.sum((predictions == labels) & (labels == label))
        total_predictions = np.sum(labels == label)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        accuracies.append(accuracy)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_labels), accuracies, color='blue')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Each Class')
    plt.xticks(range(num_labels), class_names, rotation='vertical')
    plt.tight_layout()
    plt.savefig(f'{root}/label_accuracies_{sufix}.png')
    plt.cla()


def evaluate(
        model: nn.Module, 
        valid_loader: DataLoader, 
        loss_fn: nn.Module,
        device: str,
        description: str = ""
    ) -> None:
    """
    Evaluates the given model using the provided data loader and loss function.

    Args:
        model (nn.Module): The neural network model to be validated.
        valid_loader (DataLoader): The data loader containing the validation dataset.
        loss_fn (nn.Module): The loss function used to compute the validation loss (not used for backpropagation)
        device (str): The device on which the model and data should be processed ('cuda' or 'cpu').
        description (str, optional): Additional information for tracking epoch description during training. Defaults to "".

    Returns:
        None
    """
    model.eval()
    pbar = tqdm(valid_loader, desc=description, total=len(valid_loader))
    hits = count = 0 # auxiliary variables for computing accuracy
    
    predictions_final = []
    labels_final = []

    for batch in pbar:
        # Gather batch and move to device
        video, labels = batch['clips'], batch['labels'].to(device)

        predictions_batch = []
        for clips in video: #clips es una llista de els clips d1 video
            clips = clips.to(device)
            # Forward pass
            with torch.no_grad():
                outputs = model(clips)

                output_classes = outputs.argmax(dim=1).cpu().numpy() #llista del predit per cada clip
                prediction = max(set(output_classes), key=list(output_classes).count)
                predictions_batch.append(prediction)

        hits_iter = torch.eq(torch.tensor(predictions_batch).to(device), labels).sum().item()
        hits += hits_iter
        count += len(labels)
        
        # Update progress bar with metrics
        pbar.set_postfix(
            acc=(float(hits_iter) / len(labels)),
            acc_mean=(float(hits) / count)
        )

        predictions_final.append(predictions_batch)
        labels_final.append(labels.cpu().numpy())

    acc_mean = float(hits) / count
    
    return acc_mean, predictions_final, labels_final


def create_datasets(
        frames_dir: str,
        annotations_dir: str,
        split: HMDB51Dataset.Split,
        clip_length: int,
        crop_size: int,
        temporal_stride: int,
        nt: int,
        dataset_type: int,
        ns: int
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
    print(dataset_type)
    if dataset_type == 31:
        datasets = {}
        for regime in HMDB51Dataset.Regime:
            datasets[regime.name.lower()] = HMDB51Dataset31(
                frames_dir,
                annotations_dir,
                split,
                regime,
                clip_length,
                crop_size,
                temporal_stride,
                nt      
            )
    elif dataset_type == 32:
        datasets = {}
        for regime in HMDB51Dataset.Regime:
            datasets[regime.name.lower()] = HMDB51Dataset32(
                frames_dir,
                annotations_dir,
                split,
                regime,
                clip_length,
                crop_size,
                temporal_stride,
                nt,
                ns
            )
    
    return datasets


def create_dataloaders(
        datasets: Dict[str, HMDB51Dataset],
        batch_size: int,
        batch_size_eval: int = 8,
        num_workers: int = 2,
        pin_memory: bool = True
    ) -> Dict[str, DataLoader]:
    """
    Creates data loaders for training, validation, and testing datasets.

    Args:
        datasets (Dict[str, HMDB51Dataset]): A dictionary containing datasets for training, validation, and testing.
        batch_size (int, optional): Batch size for the data loaders. Defaults to 8.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 2.
        pin_memory (bool, optional): Whether to pin memory in DataLoader for faster GPU transfer. Defaults to True.

    Returns:
        Dict[str, DataLoader]: A dictionary containing data loaders for training, validation, and testing datasets.
    """
    dataloaders = {}
    for key, dataset in datasets.items():
        dataloaders[key] = DataLoader(
            dataset,
            batch_size=(batch_size if key == 'training' else batch_size_eval),
            shuffle=(key == 'training'),  # Shuffle only for training dataset
            collate_fn=dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
            
    return dataloaders


def create_optimizer(optimizer_name: str, parameters: Iterator[nn.Parameter], lr: float = 1e-4) -> torch.optim.Optimizer:
    """
    Creates an optimizer for the given parameters.
    
    Args:
        optimizer_name (str): Name of the optimizer (supported: "adam" and "sgd" for now).
        parameters (Iterator[nn.Parameter]): Iterator over model parameters.
        lr (float, optional): Learning rate. Defaults to 1e-4.

    Returns:
        torch.optim.Optimizer: The optimizer for the model parameters.
    """
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer name: {optimizer_name}")


def print_model_summary(
        model: nn.Module,
        clip_length: int,
        crop_size: int,
        print_model: bool = True,
        print_params: bool = True,
        print_FLOPs: bool = True
    ) -> None:
    """
    Prints a summary of the given model.

    Args:
        model (nn.Module): The model for which to print the summary.
        clip_length (int): Number of frames of the clips.
        crop_size (int): Size of spatial crops (squares).
        print_model (bool, optional): Whether to print the model architecture. Defaults to True.
        print_params (bool, optional): Whether to print the number of parameters. Defaults to True.
        print_FLOPs (bool, optional): Whether to print the number of FLOPs. Defaults to True.

    Returns:
        None
    """
    if print_model:
        print(model)

    if print_params:
        num_params = sum(p.numel() for p in model.parameters())
        #num_params = model_analysis.calculate_parameters(model) # should be equivalent
        print(f"Number of parameters (M): {round(num_params / 10e6, 2)}")

    if print_FLOPs:
        num_FLOPs = model_analysis.calculate_operations(model, clip_length, crop_size, crop_size)
        print(f"Number of FLOPs (G): {round(num_FLOPs / 10e9, 2)}")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a video classification model on HMDB51 dataset.')
    parser.add_argument('frames_dir', type=str, 
                        help='Directory containing video files')
    parser.add_argument('--annotations-dir', type=str, default="data/hmdb51/testTrainMulti_601030_splits",
                        help='Directory containing annotation files')
    parser.add_argument('--clip-length', type=int, default=4,
                        help='Number of frames of the clips')
    parser.add_argument('--crop-size', type=int, default=182,
                        help='Size of spatial crops (squares)')
    parser.add_argument('--temporal-stride', type=int, default=12,
                        help='Receptive field of the model will be (clip_length * temporal_stride) / FPS')
    parser.add_argument('--model-name', type=str, default='x3d_xs',
                        help='Model name as defined in models/model_creator.py')
    parser.add_argument('--load-pretrain', action='store_true', default=False,
                    help='Load pretrained weights for the model (if available)')
    parser.add_argument('--optimizer-name', type=str, default="adam",
                        help='Optimizer name (supported: "adam" and "sgd" for now)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for the training data loader')
    parser.add_argument('--batch-size-eval', type=int, default=16,
                        help='Batch size for the evaluation data loader')
    parser.add_argument('--validate-every', type=int, default=5,
                        help='Number of epochs after which to validate the model')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of worker processes for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training (cuda or cpu)')
    parser.add_argument('--patience', type=int, default=3,
                        help='Patience value for the early stopping')
    parser.add_argument('--nt', type=int, default=5,
                        help = "Number of clips to sample from each video")
    parser.add_argument('--ns', type=int, default=3,
                        help = "Number of different crops to sample from each clip")
    
    args = parser.parse_args()

    root = 'experimentation'

    if not os.path.exists(root):
        os.makedirs(root)

    weights_folder = 'baseline_model'
    dataset_type = 32

    actual_folder = f'{root}/{dataset_type}'
    if not os.path.exists(actual_folder):
        os.makedirs(actual_folder)

    configuration['dataset_type'] = dataset_type
    configuration['weights_folder'] = weights_folder
    configuration['args'] = args
    configuration['root'] = actual_folder

    root = configuration['root']
    args = configuration['args']
    dataset_type = configuration['dataset_type']
    weights_folder = configuration['weights_folder'] 

    nombre_run = 'try'
    folder_evaluation = f'{root}/{nombre_run}'
    if not os.path.exists(folder_evaluation):
        os.makedirs(folder_evaluation)

    datasets = create_datasets(
        frames_dir=args.frames_dir,
        annotations_dir=args.annotations_dir,
        split=HMDB51Dataset.Split.TEST_ON_SPLIT_1, # hardcoded
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        temporal_stride=args.temporal_stride,
        nt=args.nt,
        dataset_type=dataset_type,
        ns=args.ns
    ) 

    # Create data loaders
    loaders = create_dataloaders(
        datasets,
        args.batch_size,
        batch_size_eval=args.batch_size_eval,
        num_workers=args.num_workers
    )

    for batch in loaders['training']:
        print(len(batch))
