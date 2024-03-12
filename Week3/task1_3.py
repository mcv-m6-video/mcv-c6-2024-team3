from __future__ import print_function

from tqdm import tqdm
from sort import Sort
from sort_OF import Sort_OF
import time
import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt  
from PIL import Image
import re
from pyflow import pyflow
import pickle
from task1_1_modif import block_matching



# SORT code provided by https://github.com/telecombcn-dl/2017-persontyle/blob/master/sessions/tracking/tracking_kalman.ipynb
# Pyflow code provided by https://github.com/pathak22/pyflow/tree/master
# Thank you all!


"""
Use Optical flow to improve the tracking from previous weeks
Most of the code is recycled from last week
- we start with the one we got the best results last week (the best combination):
    - yolov9 finetuned + sort
    - comparar amb la notra implementacio del optical flow i la millor de les ja implementades (de moment pyflow)

- calculate execution time
"""




class Tracking:

    def __init__(self, pathDets, pathImgs,pathOutput,w = 1920, h = 1080, num_files = 0, display = False, ini_0 = True, classes = [0,1,2,3,5,7], conf_thr = 0.5, optical_flow = "our_own", type_combi = "indiv"):
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
        self.optical_flow = optical_flow
        self.type_combi = type_combi #type of movement of the optical flow
        self.detections_sort = self.yolo2sort()


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
    


    def sort2mot_challenge(self, output, tracker):
        """
        Saves a txt file in pathOutput with the tracking in the format necessary to run TrackEval
        Output: frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1
        Input: frame, x1,y1,x2,y2, id
        """
        out_path = self.pathOutout + "/tracking_OF" + tracker + ".txt"
        with open(out_path, 'w') as f:
            for line in output:
                x1, y1, x2, y2 = line[1:5]
                bb_left = min(x1, x2)
                bb_top = min(y1, y2)
                bb_w = abs(x2 - x1)
                bb_h = abs(y2 - y1)
                parsed_output = [line[0], line[5], bb_left, bb_top, bb_w, bb_h, 1, -1, -1, -1]
                f.write(','.join(map(str, parsed_output)) + '\n')

    def compute_of(self, img_cur, img_prev):
        "taken from pyflow, with all default parameters"
        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

        u, v, im2W = pyflow.coarse2fine_flow(
            img_prev, img_cur, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
            nSORIterations, colType)

        flow = np.concatenate((u[..., None], v[..., None]), axis=2)
        # en toeria primer es x i despres y el moviment
 

        return flow
    
    def save_of(self, image_name, flow):
        directory = os.path.join(self.pathOutout, 'optical_flow')
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(self.pathOutout + "/optical_flow/" + image_name + '_of.pkl', 'wb') as f:
            pickle.dump(flow, f)



    def SORT_OF(self):
        # performs tracking using SORT + optical flow computed with whatever thing
        total_time = 0.0
        total_frames = 0
        out = np.array([])
        images_list = os.listdir(self.pathImgs) # we are ssuming only txt files here
        images_list = sorted(images_list)
        
        if self.optical_flow != None:
            mot_tracker = Sort_OF(type_combi=self.type_combi)
        else:
            mot_tracker = Sort()

        for frame in tqdm(range(int(self.detections_sort[:,0].max()))): # all frames in the sequence
            frame += 1 #detection and frame numbers begin at 1
            dets = self.detections_sort[self.detections_sort[:,0]==frame, -4:]   

            total_frames += 1
            start_time = time.time()

            #compute optical flow --> entenc que el of es compute amb la actual i lanterior, llavors el primer frame no en tindra, posarem dues vegades la mateixa
            if self.optical_flow == "pyflow" and frame != 1:

                img_cur = np.array(Image.open(self.pathImgs + "/" + images_list[frame-1]))
                img_cur = img_cur.astype(float) / 255.
                img_pre = np.array(Image.open(self.pathImgs + "/" + images_list[frame-2]))
                img_pre = img_pre.astype(float) / 255.

                actual_optical_flow = self.pathOutout + "/optical_flow/" + images_list[frame-1].split('.')[0] + '_of.pkl'

                if os.path.exists(actual_optical_flow):
                    with open(actual_optical_flow, 'rb') as f:
                        pred_flow = pickle.load(f)
                        
                else:
                    pred_flow = self.compute_of(img_cur, img_pre)
                    self.save_of(images_list[frame-1].split('.')[0], pred_flow)

                #we use an edited version of the sort algorithm that takes int account the flow
                trackers = mot_tracker.update(dets, pred_flow)


            elif self.optical_flow == "our_own" and frame != 1:
                path_cur = self.pathImgs + "/" + images_list[frame-1]
                path_pre = self.pathImgs + "/" + images_list[frame-2]
                actual_optical_flow = self.pathOutout + "/optical_flow_iker/" + images_list[frame-1].split('.')[0] + '_of.pkl'

                if os.path.exists(actual_optical_flow):
                    with open(actual_optical_flow, 'rb') as f:
                        pred_flow = pickle.load(f)
                        
                else:
                    pred_flow = block_matching(path_cur, path_pre)
                    self.save_of(images_list[frame-1].split('.')[0], pred_flow)

                #we use an edited version of the sort algorithm that takes int account the flow
                trackers = mot_tracker.update(dets, pred_flow)
            

            else: #dont use the optical flow
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
    


if __name__ == '__main__':
    # Read detections - one file for each image
    det_path = '/home/user/Documents/MASTER/C6/mcv-c6-2024-team3/Week2/data_yolo/dets'
    path_imgs = '/home/user/Documents/MASTER/C6/mcv-c6-2024-team3/Week2/data_yolo/images'
    path_output = '/home/user/Documents/MASTER/C6/mcv-c6-2024-team3/Week3/tracking'
    image = cv2.imread(path_imgs + "/frame00000.png")
    files = os.listdir(path_imgs)
    num_files = len(files)



    # Get the height and width
    h, w, _ = image.shape
    tracking = Tracking(det_path, path_imgs, path_output, w, h,num_files, display = False, ini_0 = True, classes = [0,1,2,3,5,7], conf_thr = 0.5, optical_flow = "our_own", type_combi="mean")
    tracking.SORT_OF()
