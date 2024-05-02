# Week 7: Multimodal

## Folder structure 
The code of the tasks and data is structured as follows:

        .
        ├── Task1                           # In this folder you can find the code of all the tasks mentioned on the folder.
        │   ├── datasets/                    # This folder remain the same as the one given.
        │   |   └── HMDB51Dataset.py         # This file has been modified to adapt the dataset to the different possible approaches.
        │   ├── models/
        │   │   └── model_creator.py         # This file has been changed to add the modifications to X3D-XS and MoViNet-A0.
        │   ├── utils/                       # We have added 2 new files
        │   |   ├── convert_to_3D.py         # It takes the npy files containing the OF and created their visualization.
        │   │   └── duplicate_last_frame.py  # This duplicates the last frame of every action.
        │   ├── pytorchtools.py              # Library for the [EarlyStop](https://github.com/Bjarten/early-stopping-pytorch/tree/master).
        │   └── train_optimize_baseline.py   # This is the baseline training code slightly modified for our task.
        ├── Task2                            # In this folder you can find the code of the task 2.
        |   ├── utils/                       # This folder remain the same as the one given.
        │   |   ├── __init__.py              # This file remain the same as the one given.
        |   |   ├── model_analysis.py        # This file remain the same as the one given.
        │   |   └── statistics.py            # This file remain the same as the one given.
        |   ├── HMDB51Dataset.py             # This file is the one where we defined the dataset class, custom for the multimodal this week.
        |   ├── data_utils.py                # A backup before implementing the TSN for training and inference :)
        │   ├── model_task1.py               # Model generator function for task 1.
        │   ├── model_task2.py               # Model generator function for task 2.
        │   ├── pytorchtools.py              # Library for the [EarlyStop](https://github.com/Bjarten/early-stopping-pytorch/tree/master).
        │   ├── task_1.py                    # Code to train the model the task 1.
        │   └── task_2.py                    # Code to train the model the task 2.
        └── README.md                        # README.md

Other files in the repository are just requirements and functions to execute the tasks.

## Running the code
To run the code you have just to add the frames folder into the root folder of the repository and then execute

```bash
python Task1/train_optimize_baseline.py
 ```

```bash
python Task2/task_X.py
 ```

## Tasks
- **Task 1:** Choose and evaluate a secondary modality by itself --> OF.
- **Task 2:** Exploit the two modalities in a multi-modal approach to try to improve unimodal models.
  - **2.1** Class score-level fusion.
  - **2.2** Early fusion.

