""" Main script for training a video classification model on HMDB51 dataset. 

Early stopping code from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb

"""

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Iterator

from torch.utils.data import DataLoader

from data_utils import MyDataset
from utils import model_analysis
from utils import statistics
import numpy as np

from pytorchtools import EarlyStopping

import matplotlib.pyplot as plt

import os
import pickle

import wandb

#------------------------
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
#------------------------

def train(
        model: nn.Module,
        train_loader: DataLoader, 
        optimizer: torch.optim.Optimizer, 
        loss_fn: nn.Module,
        device: str,
        description: str = ""
    ) -> list:
    """
    Trains the given model using the provided data loader, optimizer, and loss function.

    Args:
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): The data loader containing the training dataset.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.
        loss_fn (nn.Module): The loss function used to compute the training loss.
        device (str): The device on which the model and data should be processed ('cuda' or 'cpu').
        description (str, optional): Additional information for tracking epoch description during training. Defaults to "".

    Returns:
        None
    """
    loss_train = []
    model.train()
    pbar = tqdm(train_loader, desc=description, total=len(train_loader))
    loss_train_mean = statistics.RollingMean(window_size=len(train_loader))
    hits = count = 0 # auxiliary variables for computing accuracy
    for (X, y) in pbar:
        # Gather batch and move to device
        clips, labels = X.to(device), y.to(device)
        # Forward pass
        outputs = model(clips)
        if len(outputs) == 2:
            outputs = outputs[0]
        # Compute loss
        loss = loss_fn(outputs, labels)
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Update progress bar with metrics
        loss_iter = loss.item()

        loss_train.append(loss_iter)
        
        hits_iter = torch.eq(outputs.argmax(dim=1), labels).sum().item()
        hits += hits_iter
        count += len(labels)
        pbar.set_postfix(
            loss=loss_iter,
            loss_mean=loss_train_mean(loss_iter),
            acc=(float(hits_iter) / len(labels)),
            acc_mean=(float(hits) / count)
        )

    acc_mean = float(hits) / count
    loss_mean = np.mean(loss_train)
    
    return acc_mean, loss_mean

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
    plt.title(f'Accuracy of Each Class {sufix}')
    plt.xticks(range(num_labels), class_names, rotation='vertical')
    plt.tight_layout()
    plt.savefig(f'{root}/label_accuracies_{sufix}.png')
    plt.cla()


def evaluate(
        model: nn.Module, 
        valid_loader: MyDataset, 
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
    loss_valid_mean = statistics.RollingMean(window_size=len(valid_loader))
    hits = count = 0 # auxiliary variables for computing accuracy

    loss_valid = []

    results_predicted = []
    labels_og = []
    
    for (X, y) in pbar:
        # Gather batch and move to device
        clips, labels = X.to(device), y.to(device)
        # Forward pass
        with torch.no_grad():
            outputs = model(clips)
            if len(outputs) == 2:
                outputs = outputs[0]
            # Compute loss (just for logging, not used for backpropagation)
            loss = loss_fn(outputs, labels) 
            # Compute metrics
            loss_iter = loss.item()

            loss_valid.append(loss_iter)
            
            hits_iter = torch.eq(outputs.argmax(dim=1), labels).sum().item()
            hits += hits_iter
            count += len(labels)

            results_predicted.append(outputs.argmax(dim=1).cpu().numpy())
            labels_og.append(labels.cpu().numpy())
            
            # Update progress bar with metrics
            pbar.set_postfix(
                loss=loss_iter,
                loss_mean=loss_valid_mean(loss_iter),
                acc=(float(hits_iter) / len(labels)),
                acc_mean=(float(hits) / count)
            )

    acc_mean = float(hits) / count
    loss_mean = np.mean(loss_valid)
    return acc_mean, loss_mean, results_predicted, labels_og


def create_datasets(
        frames_dir: str,
        annotations_dir: str,
        split: MyDataset.Split,
        transform
) -> Dict[str, MyDataset]:
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
    for regime in MyDataset.Regime:
        datasets[regime.name.lower()] = MyDataset(
            videos_dir = frames_dir,
            annotations_dir = annotations_dir,
            split = split,
            regime = regime,
            transform = transform
        )
    
    return datasets


def create_dataloaders(
        datasets: Dict[str, MyDataset],
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
        num_FLOPs = model_analysis.calculate_operations(model, 1, crop_size, crop_size)
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

def get_model(name_model):
    if name_model == 'resnet152':
        
        weights = ResNet152_Weights.DEFAULT
        model = resnet152(weights=weights)
        model.fc = nn.Linear(2048, len(MyDataset.CLASS_NAMES))
        transform = weights.IMAGENET1K_V2.transforms()

    elif name_model == 'inception_v3':
        transform = Inception_V3_Weights.DEFAULT.transforms()
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.fc = nn.Linear(2048, len(MyDataset.CLASS_NAMES))
    else:
        raise ValueError(f"Unknown model name: {name_model}")
    
    return model, transform

def caller(config = None):
    with wandb.init(config=config):
        
        if not os.path.exists('experiments'):
            os.makedirs('experiments')
            
        config = wandb.config
        nombre_run = 'experiments/' + wandb.run.name
        
        #Init model, optimizer, and loss function
        model, transform = get_model(config.name_model)
        
        if not os.path.exists(nombre_run):
            os.makedirs(nombre_run)

        # Create datasets
        datasets = create_datasets(
            frames_dir='../framesgood',
            annotations_dir='data/hmdb51/testTrainMulti_601030_splits',
            split=MyDataset.Split.TEST_ON_SPLIT_1, # hardcoded
            transform=transform
        )

        # Create data loaders
        loaders = create_dataloaders(
            datasets,
            config.batch_size,
            batch_size_eval=5,
            num_workers=2
        )

        # Init optimizer, and loss function        
        optimizer = create_optimizer(config.optimizer_name, model.parameters(), lr=config.lr)
        loss_fn = nn.CrossEntropyLoss()

        # print_model_summary(model, config.crop_size)

        model = model.to('cuda')

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=4, verbose=True)

        loss_train_list = []
        acc_train_list = []
        
        loss_val_list = []
        acc_val_list = []
        epochs = []
        
        validate_every = 5
        
        for epoch in range(config.epochs):
            # We swap the order of the training and validation steps!

            # Training
            description = f"Training [Epoch: {epoch+1}/{config.epochs}]"
            acc_train, loss_train = train(model, loaders['training'], optimizer, loss_fn, 'cuda', description=description)

            loss_train_list.append(loss_train)
            acc_train_list.append(acc_train)

            wandb.log({
                'acc_train': acc_train,
                'loss_train': loss_train
            })

            # Validation
            
            if epoch % validate_every == 0:
                description = f"Validation [Epoch: {epoch+1}/{config.epochs}]"
                acc_val, loss_val, _, _ = evaluate(model, loaders['validation'], loss_fn, 'cuda', description=description)

                loss_val_list.append(loss_val)
                acc_val_list.append(acc_val)
                epochs.append(epoch)
                
                early_stopping(loss_val, model)
                if early_stopping.early_stop:
                    print("Early stopping at epoch ", epoch)
                    break

        torch.save(model.state_dict(), f'{nombre_run}/model_default.pth')

        # Testing
        acc_val_final, loss_val_final, val_predictions, gt_labels_val = evaluate(model, loaders['validation'], loss_fn, 'cuda', description=f"Validation [Final]")
        acc_test_final, loss_test_final, test_predictions, gt_label_test = evaluate(model, loaders['testing'], loss_fn, 'cuda', description=f"Testing")

        val_predictions = np.concatenate(val_predictions)
        test_predictions = np.concatenate(test_predictions)
        gt_labels_val = np.concatenate(gt_labels_val)
        gt_label_test = np.concatenate(gt_label_test)

        with open(f'{nombre_run}/val_predictions.pkl', 'wb') as f:
            pickle.dump(val_predictions, f)
        
        with open(f'{nombre_run}/test_predictions.pkl', 'wb') as f:
            pickle.dump(test_predictions, f)

        with open(f'{nombre_run}/gt_labels_train.pkl', 'wb') as f:
            pickle.dump(gt_labels_val, f)

        with open(f'{nombre_run}/gt_label_val.pkl', 'wb') as f:
            pickle.dump(gt_label_test, f)

        plot_label_accuracies(val_predictions, gt_labels_val, CLASS_NAMES, 'val', nombre_run)
        plot_label_accuracies(test_predictions, gt_label_test, CLASS_NAMES, 'test', nombre_run)

        # Test accuracy
        print("Val accuracy:", acc_val_final)
        print("Val loss:", loss_val_final)
        
        print("Test accuracy:", acc_test_final)
        print("Test loss:", loss_test_final)

        wandb.log({
            'acc_val': acc_val_final,
            'acc_test': acc_test_final
        })


sweep_config = {
    'method': 'bayes',
    'metric': {'goal': 'maximize', 'name': 'acc_val'},
    'parameters': {
        'epochs': {
            'values':[1]

        },
        'batch_size': {
            'values': [8, 16, 32, 64, 128]
        },
        'optimizer_name': {
            'values': ['adam', 'sgd']
        },
        'lr': {
            'distribution': 'uniform',
            'max': 0.0001,
            'min': 0.00001
        },
        'name_model': {
            'values': ['resnet152', 'inception_v3']
        }
    }
}

if __name__ == "__main__":

    sweep_id = wandb.sweep(sweep_config, project="ImageModels-v1")
    wandb.agent(sweep_id, function=caller, count=200)
    
    

    
