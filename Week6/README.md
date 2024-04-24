# Week 6: Action detection

## Folder structure 
The code of the tasks and data is structured as follows:

        .
        ├── task_1                           # In this folder you can find the code of all the tasks mentioned on the folder.
        │   ├── datasets/                    # This folder remain the same as the one given.
        │   |   └── HMDB51Dataset.py         # This file remain the same as the one given.
        │   ├── models/
        │   │   └── model_creator.py         # This file has been changed to add the modifications to X3D-XS and MoViNet-A0
        │   ├── utils/                       # This folder remain the same as the one given.
        │   ├── pytorchtools.py              # Library for the [EarlyStop](https://github.com/Bjarten/early-stopping-pytorch/tree/master).
        │   └── train_optimize_baseline.py   # This is the baseline training code slightly modified for our task.
        ├── Task_2                           # In this folder you can find the code of the task 2.
        |   ├── utils/                       # This folder remain the same as the one given.
        │   |   ├── __init__.py              # This file remain the same as the one given.
        |   |   ├── model_analysis.py        # This file remain the same as the one given.
        │   |   └── statistics.py            # This file remain the same as the one given.
        |   ├── data_utils.py                # This file is the one where we defined the dataset class.
        |   ├── inference.py                 # This file is used to load the best weights of the model you desire and get the metrics and plots.
        │   ├── pytorchtools.py              # Library for the [EarlyStop](https://github.com/Bjarten/early-stopping-pytorch/tree/master).
        │   └── train.py                     # This file is a wandb method to optimize the hyperparameters of the image model classification.
        └── README.md                        # Modification of util functions

Other files in the repository are just requirements and functions to execute the tasks.

## Running the code
To run the code you have just to add the frames folder into the root folder of the repository and then execute

```bash
python task_1/train_optimize_baseline.py
 ```

```bash
python Task2/SCRIPT_YOU_WANT.py
 ```

## Tasks
- **Task 1:** Changing baseline model
- **Task 2:** Temporal dynamics
