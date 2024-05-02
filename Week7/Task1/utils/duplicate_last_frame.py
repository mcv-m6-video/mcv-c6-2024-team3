import os
from tqdm import tqdm
import numpy as np

flow_dir = '/ghome/group03/C6/C6_optical_flow'

classes = sorted(os.listdir(flow_dir), reverse=True)
vids_total = 0
vids_unprocessed = 0
for i, cls in enumerate(classes):
    cls_dir = os.path.join(flow_dir, cls)
    videos = sorted(os.listdir(cls_dir), reverse=True)
    for video in tqdm(videos, desc=f'Converting class {i}/51 videos'):
        vids_total += 1
        video_dir = os.path.join(cls_dir, video)
        frames = sorted(os.listdir(video_dir))
        # if video == 'winKen_wave_u_cm_np1_ri_bad_1':
        id_num = frames[-1][5:10]
        print(os.path.join(video_dir, frames[-1]))
        last_frame = np.load(os.path.join(video_dir, frames[-1]))

        last_id = '{:05d}'.format(int(id_num)+1)
        last_id_filename = f'frame{last_id}_flow.npy'
        last_frame_copy = last_frame.copy()
        np.save(os.path.join(video_dir, last_id_filename), last_frame_copy)