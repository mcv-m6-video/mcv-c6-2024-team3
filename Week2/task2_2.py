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
import io


# SORT code provided by https://github.com/telecombcn-dl/2017-persontyle/blob/master/sessions/tracking/tracking_kalman.ipynb


"""
TO TAKE INTO ACCOUNT:
- Dsplay has not been tested, just copied


TO DO:
- kalman 
- deep sort or strongsort
- eval the results
"""

class KalmanFiltering:

    def __init__(self, pathDets, pathImgs, w = 256, h = 256, display = False):
        self.pathDets = pathDets
        self.pathImgs = pathImgs
        self.im_width = w
        self.im_height = h
        self.display = display
        self.detections = self.yolo2sort()



    def yolo2sort(self):
        files = os.listdir(self.pathDets) # we are ssuming only txt files here
        files = sorted(files)
        sort_detecs = []


        for img_detecs in files:
            file_path = os.path.join(self.pathDets, img_detecs)
            with open(file_path, 'r') as file:
                for line in file:
                    elems = [float(elem) for elem in line.split()]
                    #now we supose the name of the file is the image id
                    cat = elems[0]
                    x1 = math.floor((elems[1] - elems[3]/2) * self.im_width) # int to make it an actual pixel position
                    y1 = math.floor((elems[2] - elems[4]/2) * self.im_height)
                    x2 = math.floor((elems[1] + elems[3]/2) * self.im_width)
                    y2 = math.floor((elems[2] + elems[4]/2) * self.im_height)

                    # new format [frame, class, x1, y1, x2,y2]
                    # images start at 0, if start at 1, remove the +1
                    sort_detecs.append([int(img_detecs[:-4])+1, cat, x1, y1, x2, y2])
        
        return np.array(sort_detecs)


    def SORT(self):
        total_time = 0.0
        total_frames = 0
        out = []
        colours = np.random.rand(32,3)

        mot_tracker = Sort() #create instance of the SORT tracker



        if self.display:
            plt.ion() # for iterative display
            fig, ax = plt.subplots(1, 2,figsize=(20,20))


        for frame in range(int(self.detections[:,0].max())): # all frames in the sequence
            frame += 1 #detection and frame numbers begin at 1
            dets = self.detections[self.detections[:,0]==frame, -4:]   


            if self.display:
                fn = self.pathImgs + str(frame) + ".png" # read the frame
                im =io.imread(fn)
                ax[0].imshow(im)
                ax[0].axis('off')
                ax[0].set_title('Original Faster R-CNN detections')
                for j in range(np.shape(dets)[0]):
                    color = colours[j]
                    coords = (dets[j,0],dets[j,1]), dets[j,2]-dets[j,0], dets[j,3]-dets[j,1]
                    ax[0].add_patch(plt.Rectangle(*coords,fill=False,edgecolor=color,lw=3))


            total_frames += 1


            if self.display:
                ax[1].imshow(im)
                ax[1].axis('off')
                ax[1].set_title('Tracked Targets')


            start_time = time.time()
            trackers = mot_tracker.update(dets) # output of trackers is: similar array, where the last column is the object ID
            cycle_time = time.time() - start_time
            total_time += cycle_time

            out.append(trackers)

            for d in trackers:
                if self.display:
                    d = d.astype(np.uint32)
                    ax[1].add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))
                    ax[1].set_adjustable('box-forced')

            if self.display:
                dp.clear_output(wait=True)
                dp.display(plt.gcf())
                time.sleep(0.000001)
                ax[0].cla()
                ax[1].cla()

        print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))

        # parse output to whatever format we need
        return out




    def DeepSort(self):
        pass



    def Kalman_impl(self):
        pass




if __name__ == '__main__':
    # Read detections - one file for each image
    det_path = './dets'
    path_imgs = './imgs'

    """image = cv2.imread("your_image_file.jpg")

    # Get the height and width
    h, w = image.shape[:2]"""
    h = 256
    w = 256
    tracking = KalmanFiltering(det_path, path_imgs, w, h, display = False)

    track_res = tracking.SORT()
    print(track_res)
