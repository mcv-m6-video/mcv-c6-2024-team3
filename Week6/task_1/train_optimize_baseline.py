""" Main script for training a video classification model on HMDB51 dataset. 

Early stopping code from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb

"""

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

import os
import pickle

import wandb


def train(
        model: nn.Module,
        model_name: str,
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
    if model_name == 'movinet':
        model.clean_activation_buffers()
    pbar = tqdm(train_loader, desc=description, total=len(train_loader))
    loss_train_mean = statistics.RollingMean(window_size=len(train_loader))
    hits = count = 0 # auxiliary variables for computing accuracy
    for batch in pbar:
        # Gather batch and move to device
        clips, labels = batch['clips'].to(device), batch['labels'].to(device)
        
        outputs = model(clips)
        # Compute loss
        loss = loss_fn(outputs, labels)
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if model_name == 'movinet':
            model.clean_activation_buffers()
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
    plt.title('Accuracy of Each Class')
    plt.xticks(range(num_labels), class_names, rotation='vertical')
    plt.tight_layout()
    plt.savefig(f'{root}/label_accuracies_{sufix}.png')
    plt.cla()


def evaluate(
        model: nn.Module, 
        model_name: str,
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
    if model_name == 'movinet':
        model.clean_activation_buffers()
    pbar = tqdm(valid_loader, desc=description, total=len(valid_loader))
    loss_valid_mean = statistics.RollingMean(window_size=len(valid_loader))
    hits = count = 0 # auxiliary variables for computing accuracy

    loss_valid = []

    results_predicted = []
    labels_og = []
    
    for batch in pbar:
        # Gather batch and move to device
        clips, labels = batch['clips'].to(device), batch['labels'].to(device)
        # Forward pass
        with torch.no_grad():
            outputs = model(clips)
            # Compute loss (just for logging, not used for backpropagation)
            loss = loss_fn(outputs, labels) 
            # Compute metrics
            loss_iter = loss.item()
            
            if model_name == 'movinet':
                model.clean_activation_buffers()

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
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #num_params = model_analysis.calculate_parameters(model) # should be equivalent
        print(f"Number of parameters: {num_params}")
        print(f"Number of parameters (M): {round(num_params / 10e6, 2)}")

    if print_FLOPs:
        num_FLOPs = model_analysis.calculate_operations(model, clip_length, crop_size, crop_size)
        print(f"Number of FLOPs: {num_FLOPs}")
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


def caller(config = None):
    with wandb.init(config=config):
        
        # Set the name of the run
        wandb.run.name = 'movinet_pretrained_64_quantized_lastblockout_16frames_ts3'
        wandb.run.save()
        
        config = wandb.config
        nombre_run = wandb.run.name
        model_name = 'movinet'

        if not os.path.exists(os.path.join('results_BL', nombre_run)):
            os.makedirs(f'results_BL/{nombre_run}')

        # Create datasets
        datasets = create_datasets(
            frames_dir='/ghome/group03/C6/framesgood',
            annotations_dir='/ghome/group03/C6/MCV-M6-ActionClassificationTask/data/hmdb51/testTrainMulti_601030_splits',
            split=HMDB51Dataset.Split.TEST_ON_SPLIT_1, # hardcoded
            clip_length=config.clip_length,
            crop_size=config.crop_size,
            temporal_stride=config.temporal_stride,
        )

        # Create data loaders
        loaders = create_dataloaders(
            datasets,
            config.batch_size,
            batch_size_eval=config.batch_size,
            num_workers=1
        )

        # Init model, optimizer, and loss function
        model = model_creator.create(model_name, True, datasets["training"].get_num_classes())
        optimizer = create_optimizer(config.optimizer_name, model.parameters(), lr=config.lr)
        loss_fn = nn.CrossEntropyLoss()

        print_model_summary(model, config.clip_length, config.crop_size)

        model = model.to('cuda')

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=10, verbose=True)

        loss_train_list = []
        acc_train_list = []
        
        loss_val_list = []
        acc_val_list = []
        epochs = []
        
        for epoch in range(config.n_epochs):
            # We swap the order of the training and validation steps!

            # Training
            description = f"Training [Epoch: {epoch+1}/{config.n_epochs}]"
            acc_train, loss_train = train(model, model_name, loaders['training'], optimizer, loss_fn, 'cuda', description=description)

            loss_train_list.append(loss_train)
            acc_train_list.append(acc_train)

            # Log metrics
            wandb.log({"epoch": epoch, "training_loss": loss_train, "training_acc": acc_train})

            # Validation
            if epoch % config.validate_every == 0:
                description = f"Validation [Epoch: {epoch+1}/{config.n_epochs}]"
                acc_val, loss_val, _, _ = evaluate(model, model_name, loaders['validation'], loss_fn, 'cuda', description=description)

                loss_val_list.append(loss_val)
                acc_val_list.append(acc_val)
                epochs.append(epoch)

                # Log metrics
                wandb.log({"epoch": epoch, "validation_loss": loss_val, "validation_acc": acc_val})

                early_stopping(loss_val, model)
                if early_stopping.early_stop:
                    print("Early stopping at epoch ", epoch)
                    break

        torch.save(model.state_dict(), f'results_BL/{nombre_run}/model_default.pth')

        # Testing
        acc_train_final, loss_train_final, train_predictions, gt_labels_train = evaluate(model, model_name, loaders['validation'], loss_fn, 'cuda', description=f"Validation [Final]")
        acc_test_final, loss_test_final, test_predictions, gt_label_test = evaluate(model, model_name, loaders['testing'], loss_fn, 'cuda', description=f"Testing")

        # Log metrics
        wandb.log({"epoch": epoch, 
                   "final_validation_loss": loss_train_final, 
                   "final_validation_acc": acc_train_final, 
                   "final_test_loss": loss_test_final, 
                   "final_test_acc": acc_test_final})

        train_predictions = np.concatenate(train_predictions)
        test_predictions = np.concatenate(test_predictions)
        gt_labels_train = np.concatenate(gt_labels_train)
        gt_label_test = np.concatenate(gt_label_test)

        # if not os.path.exists('results'):
        #     os.makedirs('results')

        with open(f'results_BL/{nombre_run}/train_predictions.pkl', 'wb') as f:
            pickle.dump(train_predictions, f)
        
        with open(f'results_BL/{nombre_run}/test_predictions.pkl', 'wb') as f:
            pickle.dump(test_predictions, f)

        with open(f'results_BL/{nombre_run}/gt_labels_train.pkl', 'wb') as f:
            pickle.dump(gt_labels_train, f)

        with open(f'results_BL/{nombre_run}/gt_label_test.pkl', 'wb') as f:
            pickle.dump(gt_label_test, f)

        plot_label_accuracies(train_predictions, gt_labels_train, CLASS_NAMES, 'train', f'results_BL/{nombre_run}')
        plot_label_accuracies(test_predictions, gt_label_test, CLASS_NAMES, 'test', f'results_BL/{nombre_run}')

        # Plot train curves
        plt.plot(loss_train_list)
        plt.plot(epochs, loss_val_list)
        plt.title('Training/Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(0, 1)
        plt.legend(['Training', 'Validation'], loc='upper right')
        plt.savefig(f'results_BL/{nombre_run}/loss.png')
        plt.cla()

        # Plot accuracy curves
        plt.plot(acc_train_list)
        plt.plot(epochs, acc_val_list)
        plt.title('Training/Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.legend(['Training', 'Validation'], loc='upper right')
        plt.savefig(f'results_BL/{nombre_run}/accuracy.png')
        plt.cla()

        # Test accuracy
        print("Train accuracy:", acc_train_final)
        print("Train loss:", loss_train_final)
        
        print("Test accuracy:", acc_test_final)
        print("Test loss:", loss_test_final)


sweep_config = {
    'method': 'bayes',
    'metric': {'goal': 'maximize', 'name': 'acc_val'},
    'parameters': {
        'clip_length': {
            'values': [4], 
        },
        'n_epochs': {
            'values':[50]

        },
        'crop_size' : {
            'values': [182]
            },
        'batch_size': {
            'values': [16]
        },
        'optimizer_name': {
            'values': ['adam']
        },
        'lr': {
            'values': [1e-4]
        },
        'temporal_stride': {
            'values': [12]
        },
        'validate_every': {
            'values': [1]
        }
    }
}

if __name__ == "__main__":

    sweep_id = wandb.sweep(sweep_config, project="Week6_task_1.1_movinet")
    wandb.agent(sweep_id, function=caller, count=1)
    
    

    
