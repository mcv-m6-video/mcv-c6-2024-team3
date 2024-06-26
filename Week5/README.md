# Week 5: Action detection

## Folder structure 
The code of the tasks and data is structured as follows:

        .
        ├── Task_21_31_32/src/               # In this folder you can find the code of all the tasks mentioned on the folder.
        │   ├── datasets/                    # This folder remain the same as the one given.
        │   |   ├── HMDB51Dataset.py         # This file remain the same as the one given.
        │   |   └── HMDB51DatasetCustom.py   # This file contains the code to create the multiview datasets.
        │   ├── models/                      # This folder remain the same as the one given.
        │   ├── utils/                       # This folder remain the same as the one given.
        │   ├── best_model.py                # This is the pipeline to execute the inference setting the best parameters.
        │   ├── inference.py                 # This is the code to execute the parameter optimization for the inference.
        │   ├── pytorchtools.py              # Library for the [EarlyStop](https://github.com/Bjarten/early-stopping-pytorch/tree/master).
        │   └── train.py                     # This is the code given by default to train the model.
        ├── Task_41_42                       # In this folder you can find the code of all the tasks mentioned on the folder.
        |   ├── datasets/                    # This folder remain the same as the one given.
        │   |   ├── HMDB51Dataset.py         # This file remain the same as the one given.
        |   |   ├── HMDB51Dataset_4.py       # This file contains the modified file to load the clips as in TSN.
        │   |   └── HMDB51DatasetCustom.py   # This file contains the code to create the multiview datasets.
        |   ├── models/                      # This folder remain the same as the one given.
        |   ├── utils/                       # This folder remain the same as the one given.
        │   ├── pytorchtools.py              # Library for the [EarlyStop](https://github.com/Bjarten/early-stopping-pytorch/tree/master).
        │   └── train_optimize.py            # This code has been modified to apply the TSN method.
        └── README.md                        # Modification of util functions

Other files in the repository are just requirements and functions to execute the tasks.

## Running the code
To run the code you have just to add the frames folder into the root folder of the repository and then execute

```bash
python Task21_31_32/src/SCRIPT.py frames/
 ```

```bash
python Task_41_42/train_optimize.py
 ```

## Requirements
We add a requierements.txt file to install the libraries

## Tasks
- **Task 2:** Baseline model
  - **Task 2.1:** Run the baseline model.
- **Task 3:** Multi View
  - **Task 3.1:** Inference creating clips
  - **Task 3.2:** Inference creating clips and spatial modifications.
- **Task 4:** Improving results
  - **Task 4.1:** Implement TSN
  - **Task 4.2:** Apply any other change to improve the results
