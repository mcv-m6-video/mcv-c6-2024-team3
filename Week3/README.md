# Week 3: Video Surveillance for Road Traffic Monitoring

## Folder structure 
The code of the tasks and data is structured as follows:

        .
        ├── S01Track/                        # In this folder you can find the results of the tracking on each video of the S01 sequence.
        ├── S03Track/                        # In this folder you can find the results of the tracking on each video of the S03 sequence.
        ├── S04Track/                        # In this folder you can find the results of the tracking on each video of the S04 sequence.
        ├── tracking_block_matching/         # In this folder you can find the results of the tracking using different techniques with the block matching method.
        ├── tracking_yolo_sort/              # In this folder you can find the results of the tracking using different techniques with the optical flow.
        ├── sort_OF.py                       # A modification of SORT to run with the optical flow.
        ├── task1_1_modif.py                 # A modification of the task1_1 model to run with several images.
        ├── task1_1.py                       # The task1_1 implementation.
        ├── task1_3.py                       # The task1_3 implementation.
        ├── task1_3_evaluator.py             # The task1_3 implementation to obtain the evaluation automatically.
        ├── task1_3_runner.py                # The task1_3 implementation to get the optical flow tracking.
        ├── task1_3_visualization.py         # The task1_3 implementation to get the optical flow tracking visualization.
        ├── task2_evaluator.py               # The task2 implementation to obtain the evaluation automatically.
        ├── task2_runner.py                  # The task2 implementation to get the predictions.
        ├── task2_tracking.py                # The task2 implementation to get the tracking.
        ├── utils.py                         # Some utils functions
        └── utils_2.py                       # Modification of util functions

Other files in the repository are just requirements and functions to execute the tasks.

References to [YOLOv9](https://github.com/WongKinYiu/yolov9), [SORT](https://github.com/abewley/sort) and [DeepSORT](https://github.com/nwojke/deep_sort).

All the tracking evaluations have been performed with [TrackEval](https://github.com/JonathonLuiten/TrackEval).

## Running the code
Each task corresponds to a separate file named after the task. To execute them, simply specify the desired hyperparameter values within the "main" section of the respective file and run it using Python 3, as demonstrated below:

```bash
python3 task1_1.py
 ```

## Requirements
To run tracking is necessary to have the requirements of [SORT](https://github.com/abewley/sort) and [DeepSORT](https://github.com/nwojke/deep_sort).

To use the YOLOv9 model, you need to clone the [YOLOv9](https://github.com/WongKinYiu/yolov9) repository.

## Tasks
- **Task 1:** Optical Flow
  - **Task 1.1:** Optical flow with Block Matching
  - **Task 1.2:** Off-the-shelf Optical Flow
  - **Task 1.3:** Object Tracking with Optical Flow
- **Task 2:** MTSC tracking
