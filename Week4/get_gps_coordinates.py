'''
INPUT: ['S01Track', 'S03Track', 'S04Track']

OUTPUT:
[ una llista per cada sequencia
    [un diccionari per frame, on pot haver un none
        None, frame1
        {'id1' : [gps_cords, bbox], 'id2' : []}, frame2
        {}, 
        {},
    ], 
    [],
    [],
]

[
    frame1:{key: (c001, trackid), value: [gps, bbox]}
    frame2
]

frame_id, camera, track_id, gps, bbox, new_id
'''

from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from math import ceil
from PIL import Image

def read_time_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    
    time_dict = {}
    for line in lines:
        line = line.strip().split(' ')

        time_dict[line[0]] = float(line[1])

    return time_dict

def get_fps(seqs):
    if seqs == 'c015':
        return 8
    
    return 10


def get_gps_coordinates(sequence):
    if sequence not in ['S01Track', 'S03Track', 'S04Track']:
        raise ValueError('Invalid sequence name')
    
    root_path = Path(sequence)

    ai_challenge_path = Path('../AIchallenge')
    time_sequence = ai_challenge_path / 'cam_timestamp' / (sequence[:3] + '.txt')
    frame_num_sequence = ai_challenge_path / 'cam_framenum' / (sequence[:3] + '.txt')

    time_dict = read_time_file(time_sequence)

    seconds_per_frame = {}

    for sequence in time_dict:
        seconds_per_frame[sequence] = 1 / get_fps(sequence)

    data = []

    max_id_total = 0

    for camera_path in root_path.iterdir():

        max_id_local = 0
        
        calibration_file = camera_path / 'calibration.txt'
        roi = camera_path / 'roi.jpg'
        deep_sort = camera_path / 'tracking_DeepSort.txt'

        camera_id = camera_path.name

        roi_image = cv2.imread(str(roi), cv2.IMREAD_GRAYSCALE)

        with open(calibration_file, 'r') as f:
            lines = f.readlines()
            H = np.array([[float(val) for val in lines[0].strip().replace(';', ' ').split(' ')]])
            H = H.reshape((3, 3)).astype(np.float32)
            H_inv = np.linalg.inv(H)

        frame_start = ceil(time_dict[camera_id] / seconds_per_frame[camera_id])

        image_path = camera_path / 'frames' / f'frame{str(1 - 1).zfill(5)}.png'

        with Image.open(image_path) as img:
            w, h = img.size

        with open(deep_sort, 'r') as f:
            lines = f.readlines()
            for line in lines:
                frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z = line.strip().split(',')

                bbox = [float(bb_left), float(bb_top), float(bb_width), float(bb_height)]

                center = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, 1]

                center_gps = np.array(center)

                center_gps = H_inv@center_gps

                center_gps = center_gps / center_gps[2]

                gps = [center_gps[1], center_gps[0]]

                max_id_local = max(max_id_local, int(id))

                # Check wheter center bbox is inside the roi
                center_bbox = [int(center[1]), int(center[0])]

                image_path = camera_path / 'frames' / f'frame{str(int(frame) - 1).zfill(5)}.png'

                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])

                if bbox[0] + bbox[2] > w:
                    bbox[2] = w - bbox[0]

                if bbox[1] + bbox[3] > h:
                    bbox[3] = h - bbox[1]

                try:
                    if roi_image[center_bbox[0], center_bbox[1]] > 200:
                        data.append({
                            'frame': int(frame_start + int(frame)), 
                            'camera': camera_path.name, 
                            'track_id': int(id), 
                            'gps': gps, 
                            'bbox': bbox,
                            'path_name': image_path,
                            'new_id': max_id_total + int(id)
                            })
                except:
                    data.append({
                        'frame': int(frame_start + int(frame)), 
                        'camera': camera_path.name, 
                        'track_id': int(id), 
                        'gps': gps, 
                        'bbox': bbox,
                        'path_name': image_path,
                        'new_id': max_id_total + int(id)
                        })
                    
        max_id_total = max(max_id_total, max_id_local)
                

    return pd.DataFrame(data)


if __name__ == '__main__':
    sequence = "S03Track"
    df = get_gps_coordinates(sequence)

    unique_values = df['bbox']

    print(np.min(unique_values), np.max(unique_values))

    
    



