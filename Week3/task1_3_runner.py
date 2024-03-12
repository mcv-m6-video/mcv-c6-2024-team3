from numpy import save
from sympy import im
from torch import gt
from utils import save_frames, reset_folder
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

if __name__ == '__main__':

    video = False
    sequence_path = Path('../OptionalTaskW2Dataset/train/S03/c010/') # if this is the path of some frames put video False
    precomputed = True

    video_path = sequence_path / 'vdo.avi'
    path_output = './tracking_yolo_sort'
    yolo_runs = Path('../yolov9/runs/detect/exp/labels/')

    video_frames_tracking = os.path.join(path_output, 'frames')
    
    if video:
        reset_folder(path_output)
        save_frames(str(video_path), video_frames_tracking)

    # get detections

    # Execute yolov9
    if not precomputed:
        try:
            shutil.rmtree('../yolov9/runs/detect/exp/')
        except:
            pass

        os.system(f'python3 ../yolov9/detect.py --save-txt --save-conf --classes 0 1 2 3 5 7 --weights ../yolov9/weights/YOLOv9StrategyBWeights.pt --conf 0.5 --source {video_frames_tracking} --device 0')

    det_path = str(yolo_runs)

    img_path = os.path.join(video_frames_tracking, 'frame00000.png')
    
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    h, w, _ = image.shape
    files = os.listdir(video_frames_tracking)
    num_files = len(files)
    optical_flow = "pyflow"
    type_combi = "mean"
    
    tracking = Tracking(
        pathDets = str(yolo_runs), 
        pathImgs = video_frames_tracking, 
        pathOutput = path_output, 
        w = w, 
        h = h,
        num_files = num_files, 
        display = False, 
        ini_0 = True, 
        classes = [0,1,2,3,5,7], 
        conf_thr = 0.5, 
        optical_flow = optical_flow, 
        type_combi=type_combi
    )

    tracking.SORT_OF()

    
    os.system('python3 plot_bb_trail.py tracking_yolo_sort/tracking_SORT.txt ../OptionalTaskW2Dataset/train/S03/c010/vdo.avi tracking_yolo_sort/video_SORTOF.mp4')
   