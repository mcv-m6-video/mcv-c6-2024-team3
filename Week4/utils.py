import cv2
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import plotly.graph_objects as go
from typing import Tuple


def reset_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)

def save_frames(video_path, output_folder):
    # Check if the folder already exists
    reset_folder(output_folder)
    # Open the video
    video = cv2.VideoCapture(video_path)
    # Read the first frame
    success, image = video.read()
    # Counter to save the frames
    count = 0
    # While there are frames to read
    while success:
        # Save the frame
        cv2.imwrite(os.path.join(output_folder, 'frame' + f'{str(count).zfill(5)}.png'), image)
        # Read the next frame
        success, image = video.read()
        # Increase the counter
        count += 1
    # Close the video
    video.release()

# Code extracted from the C4 subject
def forward_warping(p: np.ndarray, H: np.ndarray) -> np.ndarray:

    x1, x2, x3 = H @ p.T
    return x1/x3, x2/x3

def backward_warping(p: np.ndarray, H: np.ndarray) -> np.ndarray:

    x1, x2, x3 = np.linalg.inv(H) @ np.array(p)
    return x1/x3, x2/x3

def get_transformed_corners(corners, H):
    transformed_corners = []
    for corner in corners:
        transformed_corners.append(forward_warping(corner, H))
    return transformed_corners

def apply_H(img, H, x_min, x_max, y_min, y_max):

    h, w, c = img.shape
    
    x_min, x_max,  y_min, y_max = compute_corners(img, H)

    width_canvas = 1920
    height_canvas = 1080

    X, Y = np.meshgrid(np.linspace(x_min, x_max, width_canvas), np.linspace(y_min, y_max, height_canvas))
    X_flat, Y_flat = X.flatten(), Y.flatten()

    dest_points = np.array([X_flat, Y_flat, np.ones_like(X_flat)])

    source_x, source_y = backward_warping(dest_points, H)

    source_x = np.reshape(source_x, X.shape) 
    source_y = np.reshape(source_y, Y.shape)

    out = np.zeros((int(np.ceil(height_canvas)), int(np.ceil(width_canvas)), 3))

    for i in range(c):
        out[:,:,i] = map_coordinates(img[:,:,i], [source_y, source_x])

    return np.uint8(out)

def compute_corners(img, H):
    h, w = img.shape[:2]

    tl = np.array([0,0,1])
    tr = np.array([w,0,1])
    bl = np.array([0,h,1])
    br = np.array([w,h,1])
    
    tl_rect = H@tl.T
    tr_rect = H@tr.T
    bl_rect = H@bl.T
    br_rect = H@br.T
    
    #-------------------------------------------------------------
    
    tl_rect_cartesian = (tl_rect / tl_rect[-1])[:2] 
    tr_rect_cartesian = (tr_rect / tr_rect[-1])[:2]
    bl_rect_cartesian = (bl_rect / bl_rect[-1])[:2]
    br_rect_cartesian = (br_rect / br_rect[-1])[:2]
    corners_rect_cartesian = np.asarray([tl_rect_cartesian, tr_rect_cartesian, br_rect_cartesian, bl_rect_cartesian])

    
    xs = np.array([corners_rect_cartesian[:,0]])
    ys = np.array([corners_rect_cartesian[:,1]])
    
    x_max, x_min = (np.max(xs)), (np.min(xs))
    y_max, y_min = (np.max(ys)), (np.min(ys))

    print(x_min, x_max,  y_min, y_max)
    
    return [x_min, x_max,  y_min, y_max]