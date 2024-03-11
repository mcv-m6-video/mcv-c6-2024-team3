from numpy import save
from torch import gt
from utils import save_frames
from pathlib import Path
import os
import shutil
from task1_3 import Tracking
import cv2
import subprocess

GT_PATH = 'TrackEval/data/gt/mot_challenge/custom-train/custom-01/gt/gt.txt'
SEQINFO_PATH = 'TrackEval/data/gt/mot_challenge/custom-train/custom-01/seqinfo.ini'
PREDICTED_PATH = 'TrackEval/data/trackers/mot_challenge/custom-train/PerfectTracker/data/custom-01.txt'

command = 'python TrackEval/scripts/run_mot_challenge.py --BENCHMARK custom --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL PerfectTracker --METRICS HOTA Identity --USE_PARALLEL False --NUM_PARALLEL_CORES 1 --DO_PREPROC False'

def read_number_of_frames(path):
    with open(path, "r") as file:
        lines = file.readlines()

        last_line = lines[-1]

        last_line_values = last_line.strip().split(',')

        number_of_frames = int(float(last_line_values[0]))
    
    return number_of_frames

if __name__ == '__main__':

    sequence_path = Path('../OptionalTaskW2Dataset/train/S03/c010/') # if this is the path of some frames put video False
    path_output = './tracking_yolo_sort'


    track_file = os.path.join(path_output, 'tracking_SORT.txt')
    gt_file = sequence_path / 'gt' / 'gt.txt'

    gt_frames = read_number_of_frames(gt)
    sort_frames = read_number_of_frames(track_file)

    number_of_frames = max(gt_frames, sort_frames)

    with open(SEQINFO_PATH, "w") as file:
        file.write("[Sequence]\n")
        file.write("name=custom\n")
        file.write("seqLength= {}\n".format(number_of_frames))

    shutil.copy(gt, GT_PATH)
    shutil.copy(track_file, PREDICTED_PATH)
    completed_process_deep = subprocess.run(command, shell=True, capture_output=True, text=True)

    with open(sequence_path / 'sort_eval.txt', 'w') as file:
        file.write(completed_process_deep.stdout)
        file.write(completed_process_deep.stderr)