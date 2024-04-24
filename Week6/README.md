# Week 4: Video Surveillance for Road

## Folder structure 
The code of the tasks and data is structured as follows:

        .
        ├── S01Track/                        # In this folder you can find the results of the tracking on each video of the S01 sequence.
        └── utils_2.py                       # Modification of util functions

Other files in the repository are just requirements and functions to execute the tasks.

References to [YOLOv9](https://github.com/WongKinYiu/yolov9), [SORT](https://github.com/abewley/sort), [DeepSORT](https://github.com/nwojke/deep_sort).

All the tracking evaluations have been performed with [TrackEval](https://github.com/JonathonLuiten/TrackEval).

## Running the code
Each task corresponds to a separate file named after the task. To execute them, simply specify the desired hyperparameter values within the "main" section of the respective file and run it using Python 3, as demonstrated below:

```bash
python3 task1_1.py
 ```

## Requirements
To run tracking is necessary to have the requirements of [SORT](https://github.com/abewley/sort), [DeepSORT](https://github.com/nwojke/deep_sort).

To use the YOLOv9 model, you need to clone the [YOLOv9](https://github.com/WongKinYiu/yolov9) repository.

## Tasks
- **Task 1:** Speed estimation
  - **Task 1.1:** Use the given dataset
  - **Task 1.2:** Use your own data
- **Task 2:** Multi Camera Tracking

## Linkl to the slides
https://docs.google.com/presentation/d/16eibYtT_KVRATw6F7Q0GUhfN67-a0FJRqyS7T9o2yio/edit?usp=drive_link
