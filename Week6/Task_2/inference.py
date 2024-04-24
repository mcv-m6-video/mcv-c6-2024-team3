""" Main script for training a video classification model on HMDB51 dataset. 

Early stopping code from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb

"""

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Iterator

from torch.utils.data import DataLoader

from data_utils import MyDatasetInference
from utils import model_analysis
from utils import statistics
import numpy as np

from pytorchtools import EarlyStopping

import matplotlib.pyplot as plt

import os
import pickle

import wandb

import cv2
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

#------------------------
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
#------------------------

configuration = {}

def plot_label_accuracies_pairs(predictions_labels, class_names, sufix, root):
    """
    Plot the accuracy of each label.

    Args:
    - predictions_labels (list of tuples): List of tuples containing (predicted label, true label).
    - class_names (list): List of class names corresponding to the labels.
    - sufix (str): Suffix to be appended to the filename.
    - root (str): Root directory where the plot will be saved.

    Returns:
    - None
    """
    # Extract predictions and labels from the list of tuples
    predictions, labels = zip(*predictions_labels)
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Calculate accuracy for each label
    num_labels = len(class_names)
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
    plt.title(f'Accuracy of Each Class {sufix} videos')
    plt.xticks(range(num_labels), class_names, rotation='vertical')
    plt.tight_layout()
    plt.savefig(f'{root}/label_accuracies_{sufix}_videos.png')
    plt.close()

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
    plt.title(f'Accuracy of Each Class {sufix} frames')
    plt.xticks(range(num_labels), class_names, rotation='vertical')
    plt.tight_layout()
    plt.savefig(f'{root}/label_accuracies_{sufix}_frames.png')
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
    predictions_per_video_total = []
    hits_video = 0
    count_video = 0

    for X,y in pbar:
        # Gather batch and move to device
        #video, labels = X, torch.tensor(y).to(device)
        list_length = len(X)
        labels = torch.tensor([y]*list_length).to(device)

        predictions_batch = []
        for clips in X: #each clip is a frame now of the same video --> agregate for all clips. abans cada clip era un clip (conjunt de frames)
            clips = clips.float().unsqueeze(0)
            clips = clips.to(device)
            # Forward pass
            with torch.no_grad():
                outputs = model(clips)

                if len(outputs) == 2:
                    outputs = outputs[0]
                prediction = outputs.argmax(dim=1).cpu().numpy() #llista del predit per cada clip
                predictions_batch.append(prediction[0])
        #print(len(predictions_batch),len(X))
        #print(torch.eq(torch.tensor(predictions_batch).to(device), labels).size())
        pred_per_video = max(set(predictions_batch), key=predictions_batch.count)

        
        hits_iter = torch.eq(torch.tensor(predictions_batch).to(device), labels).sum().item()
        hits += hits_iter
        count += len(labels)
        count_video += 1

        #single prediction per video
        if pred_per_video ==y:
            hits_video += 1
        
        # Update progress bar with metrics
        pbar.set_postfix(
            acc=(float(hits_iter) / len(labels)), # accuracy es per 1 video
            acc_mean=(float(hits) / count), #mean accuracy de tots els frames (dels videos)fins el moment)
            acc_mean_per_video = (float(hits_video) / count_video)
        )

        
        predictions_per_video_total.append([pred_per_video, y]) #llista en que cada posicio es la llista de la prediccio pel video i la gt
        predictions_final.append(predictions_batch) #llista en que cada posicio es la llista son totes les prediccions dels frames d1 video
        labels_final.append(labels.cpu().numpy())

    acc_mean = float(hits) / count
    
    return acc_mean, predictions_final, labels_final, predictions_per_video_total


def create_datasets(
        frames_dir: str,
        annotations_dir: str,
        split: MyDatasetInference.Split,
        transform
) -> Dict[str, MyDatasetInference]:
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
    for regime in MyDatasetInference.Regime:
        datasets[regime.name.lower()] = MyDatasetInference(
            videos_dir = frames_dir,
            annotations_dir = annotations_dir,
            split = split,
            regime = regime,
            transform = transform
        )
    
    return datasets


def create_dataloaders(
        datasets: Dict[str, MyDatasetInference],
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

def get_model(name_model):
    if name_model == 'resnet152':
        
        weights = ResNet152_Weights.DEFAULT
        model = resnet152(weights=weights)
        model.fc = nn.Linear(2048, len(MyDatasetInference.CLASS_NAMES))
        transform = weights.IMAGENET1K_V2.transforms()

    elif name_model == 'inception_v3':
        transform = Inception_V3_Weights.DEFAULT.transforms()
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.fc = nn.Linear(2048, len(MyDatasetInference.CLASS_NAMES))

    else:
        raise ValueError(f"Unknown model name: {name_model}")
    
    return model, transform

def plot_confusion_matrix_pair(predictions_labels, class_names, sufix, root):
    """
    Plot the confusion matrix.

    Args:
    - predictions_labels (list of tuples): List of tuples containing (predicted label, true label).
    - class_names (list): List of class names corresponding to the labels.
    - sufix (str): Suffix to be appended to the filename.
    - root (str): Root directory where the plot will be saved.

    Returns:
    - None
    """
    # Extract predictions and labels from the list of tuples
    predictions, labels = zip(*predictions_labels)
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Generate confusion matrix
    cm = confusion_matrix(labels, predictions)

    # Plotting confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix {sufix} videos')
    plt.tight_layout()
    plt.savefig(f'{root}/confusion_matrix_{sufix}_videos.png')
    plt.close()

def plot_confusion_matrix(predictions, labels, class_names, sufix, root):
    """
    Plot the confusion matrix.

    Args:
    - predictions (numpy.ndarray): Array of predicted labels.
    - labels (numpy.ndarray): Array of true labels.
    - class_names (list): List of class names corresponding to the labels.
    - sufix (str): Suffix to be appended to the filename.
    - root (str): Root directory where the plot will be saved.

    Returns:
    - None
    """
    # Generate confusion matrix
    cm = confusion_matrix(labels, predictions)

    # Plotting confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix {sufix} frames')
    plt.tight_layout()
    plt.savefig(f'{root}/confusion_matrix_{sufix}_frames.png')
    plt.close()

def compute_accuracy(predictions_labels):
    """
    Compute the accuracy based on a list of tuples containing predicted and ground truth labels.

    Args:
    - predictions_labels (list of tuples): List of tuples containing (predicted label, true label).

    Returns:
    - accuracy (float): Accuracy of the predictions.
    """
    correct = 0
    total = len(predictions_labels)

    for pred, gt in predictions_labels:
        if pred == gt:
            correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy

def caller():
    args = configuration['args']
    weights_folder = configuration['weights_folder'] 

    folder_evaluation = weights_folder

    model, transform = get_model(configuration['model_type'])
    
    datasets = create_datasets(
        frames_dir=args.frames_dir,
        annotations_dir=args.annotations_dir,
        split=MyDatasetInference.Split.TEST_ON_SPLIT_1, # hardcoded
        transform=transform
    ) 

    # Create data loaders
    loaders = create_dataloaders(
        datasets,
        args.batch_size,
        batch_size_eval=args.batch_size_eval,
        num_workers=args.num_workers
    )

    # Init model, optimizer, and loss function
    model.load_state_dict(torch.load(weights_folder + '/model_default.pth'))

    loss_fn = nn.CrossEntropyLoss()

    # print_model_summary(model, config.clip_length, config.crop_size)

    model = model.to(args.device)

    # Testing
    acc_val_final, predictions_val, labels_val, predictions_per_video_total_val = evaluate(model, datasets['validation'], loss_fn, args.device, description=f"Validation [Final]")
    predictions_val = np.concatenate(predictions_val)
    labels_val = np.concatenate(labels_val)

    plot_label_accuracies(predictions_val, labels_val, CLASS_NAMES, 'validation', folder_evaluation)
    plot_label_accuracies_pairs(predictions_per_video_total_val, CLASS_NAMES, 'validation', folder_evaluation)
    plot_confusion_matrix_pair(predictions_per_video_total_val, CLASS_NAMES, 'validation', folder_evaluation)
    plot_confusion_matrix(predictions_val, labels_val, CLASS_NAMES, 'validation', folder_evaluation)

    print(f"Validation accuracy: {acc_val_final}")

    acc_video_val = compute_accuracy(predictions_per_video_total_val)
    print(f"Validation accuracy per video: {acc_video_val}")

    acc_test_final, predictions_test, labels_test, predictions_per_video_total_test = evaluate(model, datasets['testing'], loss_fn, args.device, description=f"Testing")
    predictions_test = np.concatenate(predictions_test)
    labels_test = np.concatenate(labels_test)
    plot_label_accuracies(predictions_test, labels_test, CLASS_NAMES, 'testing', folder_evaluation)
    plot_label_accuracies_pairs(predictions_per_video_total_test, CLASS_NAMES, 'testing', folder_evaluation)    
    plot_confusion_matrix(predictions_test, labels_test, CLASS_NAMES, 'testing', folder_evaluation)
    plot_confusion_matrix_pair(predictions_per_video_total_test, CLASS_NAMES, 'testing', folder_evaluation)

    print(f"Testing accuracy: {acc_test_final}")

    acc_video_test = compute_accuracy(predictions_per_video_total_test)
    print(f"Testing accuracy per video: {acc_video_test}")
    # Test accuracy


sweep_config = {
    
}

def inference(video_path):
    video = cv2.VideoCapture(video_path)

    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)

    model, transform = get_model(configuration['model_type'])
    model.load_state_dict(torch.load(weights_folder + '/model_default.pth', map_location=torch.device('cpu')))
    model.eval()
    # model = model.to(args.device)

    to_tensor = transforms.ToTensor()

    classes_predictions = []
    for frame in frames:
        frame = to_tensor(frame)
        frame = frame.unsqueeze(0)
        frame = transform(frame)
        # frame = frame.to(args.device)

        with torch.no_grad():
            outputs = model(frame)
            if len(outputs) == 2:
                outputs = outputs[0]
            prediction = outputs.argmax(dim=1).cpu().numpy()
            classes_predictions.append(prediction[0])

    # Max class predictions
    max_class = max(set(classes_predictions), key=classes_predictions.count)
    print(f"Max class: {MyDatasetInference.CLASS_NAMES[max_class]}")

    # Print class predictions accuracy
    class_accuracies = []
    for i in range(len(MyDatasetInference.CLASS_NAMES)):
        correct_predictions = np.sum((np.array(classes_predictions) == i))
        total_predictions = len(classes_predictions)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        if accuracy != 0:
            class_accuracies.append(accuracy)
            print(f"Class {MyDatasetInference.CLASS_NAMES[i]}: {accuracy}")

    
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a video classification model on HMDB51 dataset.')
    parser.add_argument('--frames_dir', type=str, 
                        help='Directory containing video files', default='../framesgood')
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
    parser.add_argument('--dataset_type', type=int, default=31,
                        help = "Type of dataset to use")
    
    args = parser.parse_args()

    nombre_run = 'wild-sweep-2'
    weights_folder = 'experiments/' + nombre_run
    dataset_type = args.dataset_type

    configuration['model_type'] = 'resnet152' #resnet152
    configuration['args'] = args
    configuration['weights_folder'] = weights_folder
    # inception_v3

    
    configuration['weights_folder'] = weights_folder
    configuration['args'] = args
    '''
    folder_path = "custom_videos/"
    files = os.listdir(folder_path)
    mp4_files = [file for file in files if file.endswith(".mp4")]
    video_paths = sorted([os.path.join(folder_path, file) for file in mp4_files])

    for video_path in video_paths:
        print(video_path)
        inference(video_path)
        print('---------------------')
    '''
    caller()
    
    