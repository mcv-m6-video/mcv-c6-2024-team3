# Week 4: Video Surveillance for Road

## Folder structure 
The code of the tasks and data is structured as follows:

        .
        ├── GT_custom_videos/                # The data annotated for the task 1.2
        ├── Task1/                           # All the files for the task 1
        ├── utils/                           # Some files for testing and visualization
        ├── get_gps_coordinates.py           # Function to get the GPS coordinates from some points
        ├── refinedTracking2MOT.py           # Move the file format from the one from task 2 to mot challenge
        ├── reid_insideframe.py              # The code for compute the reidintification for task 2
        ├── task2_X.py                       # Are all the steps from previous weeks to obtain the predictions and the tracking
        └── utils.py                         # Some functions to use in some files
        

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

## Link to the slides
https://docs.google.com/presentation/d/16eibYtT_KVRATw6F7Q0GUhfN67-a0FJRqyS7T9o2yio/edit?usp=drive_link
