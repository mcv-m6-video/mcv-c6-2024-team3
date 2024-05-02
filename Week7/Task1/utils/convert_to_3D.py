import os
import cv2
import numpy as np
from tqdm import tqdm

flow_dir = '/ghome/group03/C6/C6_optical_flow'
flow_dir_3D = '/ghome/group03/C6/C6_3D_optical_flow'

def visualize_flow(src_path, write_path):
  gm_flow = np.load(src_path)
  x_gm = gm_flow[:,:,0]
  y_gm = gm_flow[:,:,1]
  flow = np.stack((x_gm, y_gm), axis = 2)
  flow = np.array(flow)
  # Use Hue, Saturation, Value colour model
  hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
  hsv[..., 2] = 255

  mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
  hsv[..., 0] = ang / np.pi / 2 * 180
  hsv[..., 1] = np.clip(mag * 255 / 24, 0, 255)
  bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
  cv2.imwrite(write_path, bgr)

classes = sorted(os.listdir(flow_dir), reverse=True)
vids_total = 0
vids_unprocessed = 0
for i, cls in enumerate(classes):
    cls_dir = os.path.join(flow_dir, cls)
    videos = sorted(os.listdir(cls_dir), reverse=True)
    if cls == 'brush_hair':
        for video in tqdm(videos[-2:], desc=f'Converting class {i}/51 videos'):
            vids_total += 1
            video_dir = os.path.join(cls_dir, video)
            dest_dir = video_dir.replace(flow_dir, flow_dir_3D)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            frames = [frame for frame in os.listdir(video_dir) if frame.endswith('npy')]
            for frame in frames:
                
                frame_path = os.path.join(video_dir, frame)
                frame_dest = frame_path.replace(flow_dir, flow_dir_3D).replace('npy','png')
                visualize_flow(frame_path, frame_dest)
            