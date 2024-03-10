import os
from pathlib import Path
import shutil
from utils import save_frames

# https://pramod-atre.medium.com/understanding-object-tracking-a-hands-on-approach-part-1-3fb1afd0ae46

if __name__ == '__main__':

    yolo_runs = Path('../yolov9/runs/detect/exp/labels/')
    
    root_path = Path('../OptionalTaskW2Dataset/train')

    for sequence in root_path.iterdir():
        
        folder_output = sequence.name + 'Track'

        try:
            os.mkdir(folder_output)
        except:
            shutil.rmtree(folder_output)
            os.mkdir(folder_output)

        for video in sequence.iterdir():
            
            text_file = str(video / 'gt' / 'gt.txt')
            video_path = str(video / 'vdo.avi')

            sequence_output = os.path.join(folder_output, video.name)
            try:
                os.mkdir(sequence_output)
            except:
                shutil.rmtree(sequence_output)
                os.mkdir(sequence_output)

            out_frames = os.path.join(sequence_output, 'frames')

            print('Creating the frames.')
            save_frames(video_path, out_frames)
            print('Frames created.')

            shutil.copy(text_file, sequence_output)

            # Execute yolov9
            os.system(f'python ../yolov9/detect.py --save-txt --save-conf --classes 0 1 2 3 5 7 --weights ../yolov9/weights/YOLOv9StrategyCWeights.pt --conf 0.5 --source {out_frames} --device 0')

            # Execute the file task_2.py
            os.system(f'python task_2.py --det_path {yolo_runs} --path_imgs {out_frames} --path_output {sequence_output}')

            shutil.rmtree('../yolov9/runs/detect/exp/')






            





