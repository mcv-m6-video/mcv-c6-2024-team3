from numpy import save
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

    video = True
    sequence_path = Path('../OptionalTaskW2Dataset/train/S03/c010/') # if this is the path of some frames put video False

    video_path = sequence_path / 'vdo.avi'
    path_output = './tracking_yolo_sort'

    reset_folder(path_output)

    video_frames_tracking = os.path.join(path_output, 'frames')

    if video:
        save_frames(str(video_path), video_frames_tracking)
    
    else:
        video_frames_tracking = video_path

    # get detections
    yolo_runs = Path('../yolov9/runs/detect/exp/labels/')

    try:
        shutil.rmtree('../yolov9/runs/detect/exp/')
    except:
        pass

    # Execute yolov9
    os.system(f'python ../yolov9/detect.py --save-txt --save-conf --classes 0 1 2 3 5 7 --weights ../yolov9/weights/YOLOv9StrategyCWeights.pt --conf 0.5 --source {video_frames_tracking} --device 0')
    
    det_path = str(yolo_runs)
    
    image = cv2.imread(os.path.join(video_frames_tracking, 'frame00000.png'))
    
    h, w, _ = image.shape
    files = os.listdir(video_frames_tracking)
    num_files = len(files)
    optical_flow = "pyflow"
    type_combi = "median"
    
    tracking = Tracking(det_path = str(yolo_runs), path_imgs = video_frames_tracking, path_output = path_output, w = w, h = h,num_files = num_files, display = False, ini_0 = True, classes = [0,1,2,3,5,7], conf_thr = 0.4, optical_flow = optical_flow, type_combi=type_combi)