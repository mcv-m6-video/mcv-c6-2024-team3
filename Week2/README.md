# Week 1: Video Surveillance for Road Traffic Monitoring

## Folder structure 
The code of the tasks and data is structured as follows:

        .
        ├── task1_2.py              # Annotations related
        ├── task2_2.py              # Tracking with SORT and DeepSORT       
        ├── task_optional.py        # CVPR 2022 AI City Challenge
        └── something else      

Other files in the repository are just requierements and funtions to execute the tasks.

References to [SORT](https://github.com/abewley/sort) and [DeepSORT](https://github.com/nwojke/deep_sort).

## Running the code
Each task corresponds to a separate file named after the task. To execute them, simply specify the desired hyperparameter values within the "main" section of the respective file and run it using Python 3, as demonstrated below:

```bash
python3 task1_1.py
 ```

## Requirements
To run Task2.2 it is necessary to have the requirements of [SORT](https://github.com/abewley/sort) and [DeepSORT](https://github.com/nwojke/deep_sort).

## Tasks
- **Task 1:** Object detection
  - **Task 1.1:** Off-the-shelf
  - **Task 1.2:** Annotation
  - **Task 1.3:** Fine-tune to your data
  - **Task 1.4:** K-Fold Cross-validation
- **Task 2:** Object tracking
  - **Task 2.1:** Tracking by Overlap
  - **Task 2.2:** Tracking with a Kalman Filter
  - **Task 2.3:** IDF1, HOTA scores
