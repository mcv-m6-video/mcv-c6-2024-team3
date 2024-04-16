# Week 5: Action detection

## Folder structure 
The code of the tasks and data is structured as follows:

        .
        ├── Task_21_31_32/src/               # In this folder you can find the results of the tracking on each video of the S01 sequence.
        │   ├── datasets/                    # This folder remain the same as the one given.
        │   ├── models/                      # This folder remain the same as the one given.
        │   ├── utils/                       # This folder remain the same as the one given.
        │   ├── best_model.py                # This is the pipeline to execute the inference setting the best parameters.
        │   ├── inference.py                 # This is the code to execute the parameter optimization for the inference.
        │   ├── pytorchtools.py              # Library for the [EarlyStop](https://github.com/Bjarten/early-stopping-pytorch/tree/master).
        │   └── train.py                     # This is the code given by default to train the model.
        └── README.md                        # Modification of util functions

Other files in the repository are just requirements and functions to execute the tasks.

## Running the code
To run the code you have just to add the frames folder into the root folder of the repository and then execute

```bash
python Task21_31_32/src/SCRIPT.py frames/
 ```

## Requirements
We add a requierements.txt file to install the libraries

## Tasks
- **Task 2:** Baseline model
  - **Task 2.1:** Run the baseline model.
- **Task 3:** Multi View
  - **Task 3.1:** Inference creating clips
  - **Task 3.2:** Inference creating clips and spatial modifications.
