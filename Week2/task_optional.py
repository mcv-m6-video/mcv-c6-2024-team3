# python detect.py --save-txt --save-conf --classes 0 1 2 3 5 7 --weights weights/yolov9-c.pt --conf 0.7 --source ../Week2/S01_c001/ --device 0

from __future__ import print_function
from sort import Sort
import time
import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt  
import matplotlib.patches as patches
from IPython import display as dp
from skimage import io
import re
from deep_sort_app_v2 import run as deep_sort_run
from utils import reset_folder



# SORT code provided by https://github.com/telecombcn-dl/2017-persontyle/blob/master/sessions/tracking/tracking_kalman.ipynb
# DEEP SORT code provided by https://github.com/nwojke/deep_sort

"""
TO TAKE INTO ACCOUNT:
- We need to optimize DeepSort hyperparameters
- Important, i changed the names of the images to be 00000.jpg instead of 00000_png(something_weird).jpg
"""

class KalmanFiltering:

    def __init__(self, pathDets, pathImgs,pathOutput,w = 1920, h = 1080, num_files = 0, display = False, ini_0 = True, classes = [0,1,2,3,5,7], conf_thr = 0.5):
        self.pathDets = pathDets
        self.pathImgs = pathImgs
        self.pathOutout = pathOutput
        self.im_width = w
        self.im_height = h
        self.display = display
        self.num_files = num_files
        self.ini_0 = ini_0
        self.classes = classes
        self.conf_thr = conf_thr
        self.detections_sort = self.yolo2sort()
        self.detection_mot = self.yolo2mot()



    def yolo2sort(self):
        files = os.listdir(self.pathDets) # we are ssuming only txt files here
        files = sorted(files)
        sort_detecs = []

        for img_detecs in files:
            file_path = os.path.join(self.pathDets, img_detecs)
            #FrameID starts at 1 --> I force it
            with open(file_path, 'r') as file:
                frame_real = int(re.findall(r'\d+', img_detecs)[0])
                if self.ini_0:
                    frame_real +=1
                for line in file:
                    elems = [float(elem) for elem in line.split()]
                    #now we supose the name of the file is the image id
                    cat = elems[0]
                    conf = elems[5]
                    if cat in self.classes and conf >= self.conf_thr:
                        x1 = math.floor((elems[1] - elems[3]/2) * self.im_width) # int to make it an actual pixel position
                        y1 = math.floor((elems[2] - elems[4]/2) * self.im_height)
                        x2 = math.floor((elems[1] + elems[3]/2) * self.im_width)
                        y2 = math.floor((elems[2] + elems[4]/2) * self.im_height)

                        # new format [frame, class, x1, y1, x2,y2]
                        sort_detecs.append([frame_real, cat, x1, y1, x2, y2])
            
        return np.array(sort_detecs)
    


    def yolo2mot(self):
        files = os.listdir(self.pathDets) # we are ssuming only txt files here
        files = sorted(files)
        mot_detecs = []
        for img_detecs in files:
            file_path = os.path.join(self.pathDets, img_detecs)
            with open(file_path, 'r') as file:
                frame_real = int(re.findall(r'\d+', img_detecs)[0])
                if self.ini_0:
                    frame_real +=1
                for line in file:
                    elems = [float(elem) for elem in line.split()]
                    cat = elems[0]
                    if cat in self.classes and elems[5] >= self.conf_thr:
                        x, y, w, h, conf = elems[1:6]
                        bb_left = x * self.im_width - w * self.im_width / 2
                        bb_top = y * self.im_height - h * self.im_height / 2
                        bb_width = w * self.im_width
                        bb_height = h * self.im_height

                        # Output: frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1
                        mot_detecs.append([frame_real, -1, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1])
        return np.array(mot_detecs)
    

    def sort2mot_challenge(self, output, tracker):
        """
        Saves a txt file in pathOutput with the tracking in the format necessary to run TrackEval
        Output: frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1
        Input: frame, x1,y1,x2,y2, id
        """
        out_path = self.pathOutout + "/tracking_" + tracker + ".txt"
        with open(out_path, 'w') as f:
            for line in output:
                x1, y1, x2, y2 = line[1:5]
                bb_left = min(x1, x2)
                bb_top = min(y1, y2)
                bb_w = abs(x2 - x1)
                bb_h = abs(y2 - y1)
                parsed_output = [line[0], line[5], bb_left, bb_top, bb_w, bb_h, 1, -1, -1, -1]
                f.write(','.join(map(str, parsed_output)) + '\n')



    def SORT(self):
        total_time = 0.0
        total_frames = 0
        out = np.array([])
        #colours = np.random.rand(32,3)

        mot_tracker = Sort() #create instance of the SORT tracker

        if self.display:
            plt.ion() # for iterative display
            fig, ax = plt.subplots(1, 2,figsize=(20,20))
            images_list = os.listdir(self.pathImgs) # we are ssuming only txt files here
            images_list = sorted(images_list)

        for frame in range(int(self.detections_sort[:,0].max())): # all frames in the sequence
            frame += 1 #detection and frame numbers begin at 1
            dets = self.detections_sort[self.detections_sort[:,0]==frame, -4:]   


            total_frames += 1


            start_time = time.time()
            trackers = mot_tracker.update(dets)# output of trackers is: similar array, where the last column is the object ID

            
            #parse a bit trackers output so it has the frame id
            frames_col = int(frame) * np.ones((trackers.shape[0], 1), dtype=trackers.dtype)
            trackers = np.hstack((frames_col, trackers))


            cycle_time = time.time() - start_time
            total_time += cycle_time
            if frame == 1:
                out = trackers
            else:
                out = np.vstack((out, trackers))

        print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))
        # parse output to mot_challenge
        self.sort2mot_challenge(out, "SORT")
    

    def DeepSORT(self):
        """
        TO DO:
        - optimize the default values
        """

        start_time = time.time()

        # detections need to be a .npy in mot challenge format
        # a bit useless step but bring save the detections to npy.
        np.save('detections_mot.npy', self.detection_mot)
        deep_sort_run(self.pathImgs, "./detections_mot.npy", self.pathOutout + "/tracking_DeepSort.txt", min_confidence = self.conf_thr, nn_budget = 100, display = self.display, nms_max_overlap=0.7, min_detection_height=30, max_cosine_distance=0.2)
        end_time = time.time()
        total_time = end_time-start_time
        print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,self.num_files,self.num_files/total_time))




if __name__ == '__main__':
    # Read detections - one file for each image
    det_path = '../yolov9/runs/detect/S01_c001_B/labels'
    path_imgs = 'S01_c001'
    path_output = './trackingS01_c001_B'
    image = cv2.imread(path_imgs + "/frame00000.png")

    files = os.listdir(path_imgs)
    num_files = len(files)

    reset_folder(path_output)

    # Get the height and width
    h, w, _ = image.shape
    tracking = KalmanFiltering(det_path, path_imgs, path_output, w, h,num_files, display = False, ini_0 = True, classes = [0,1,2,3,5,7], conf_thr = 0.5)
    tracking.DeepSORT()
    tracking.SORT()

# python plot_bb_trail.py trackingS01_c001/tracking_DeepSort.txt ../OptionalTaskW2Dataset/train/S01/c001/vdo.avi trackingS01_c001/video_DeepSort.mp4
# python detect.py --save-txt --save-conf --classes 0 1 2 3 5 7 --weights weights/yolov9-c.pt --conf 0.001 --source ../Week2/S01_c001/ --device 0