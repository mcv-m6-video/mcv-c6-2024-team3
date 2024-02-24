import cv2
import numpy as np


def task1_1(seq, alpha = 2):
    # split 25% --> train
    total_frames = int(seq.get(cv2.CAP_PROP_FRAME_COUNT))
    train_25 = int(total_frames * 0.25)

    # to make it computationally efficient, stack all frames and compute mean and variance at once
    frames = []
    frame_count = 0
    while seq.isOpened() and frame_count < train_25:
        
        _, frame = seq.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frames.append(gray_frame)

        frame_count += 1

    seq.release()
    stacked_frames = np.stack(frames, axis=0)

    # Compute the mean and variance for each pixel
    means= np.mean(stacked_frames, axis=0)
    variances = np.var(stacked_frames, axis=0)

    remaining_frames = total_frames - train_25


    """
    Classificate the pixels of the remaining frames of the video
    """
    # start reading the sequence where we left it
    seq.set(cv2.CAP_PROP_POS_FRAMES, train_25)

    rest_frames = []

    # Read and process remaining frames
    while seq.isOpened():
        _, frame = seq.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rest_frames.append(gray_frame)
    
    seq.release()
    rest_frames = np.stack(rest_frames, axis=0)

    # calculate if each pixel belongs to foreground or background

    





if __name__ == '__main__':
    path = path = '/home/user/Documents/MASTER/C6/AICity_data/train/S03/c010/vdo.avi'
    seq = cv2.VideoCapture(path)
    alpha = 0.2 # change this parameter
    m, var, rem_frames = task1_1(seq, alpha)