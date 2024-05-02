import os
import cv2
import numpy as np
from tqdm import tqdm

flow_dir = '/home/georg/C6_optical_flow/'
flow_dir_padded = '/home/georg/C6_zero_padded_optical_flow/'

def zero_pad_flow(src_path, write_path):
  gm_flow = np.load(src_path)
  x_gm = gm_flow[:,:,0]
  y_gm = gm_flow[:,:,1]
  zeroes = np.zeros_like(x_gm)
  padded = np.stack((x_gm, y_gm, zeroes), axis = 2)
  np.save(write_path, padded)

classes = sorted(os.listdir(flow_dir), reverse=True)
vids_total = 0
vids_unprocessed = 0
for i, cls in enumerate(classes):
    cls_dir = os.path.join(flow_dir, cls)
    videos = sorted(os.listdir(cls_dir), reverse=True)
    for video in tqdm(videos, desc=f'Converting class {i}/51 videos'):
        vids_total += 1
        video_dir = os.path.join(cls_dir, video)
        dest_dir = video_dir.replace(flow_dir, flow_dir_padded)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        frames = os.listdir(video_dir)
        for frame in frames:
            frame_path = os.path.join(video_dir, frame)
            frame_dest = frame_path.replace(flow_dir, flow_dir_padded)
            zero_pad_flow(frame_path, frame_dest)
            