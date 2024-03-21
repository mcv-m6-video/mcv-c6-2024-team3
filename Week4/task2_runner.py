import os
from pathlib import Path
import shutil
from utils import save_frames

if __name__ == '__main__':

    yolo_runs = Path('../yolov9/runs/detect/exp/labels/')
    
    root_path = Path('../AIchallenge/train')

    for sequence in root_path.iterdir():
        
        folder_output = sequence.name + 'Track'

        try:
            os.mkdir(folder_output)
        except:
            pass

        for video in sequence.iterdir():
            
            text_file = str(video / 'gt' / 'gt.txt')
            video_path = str(video / 'vdo.avi')
            calibration = str(video / 'calibration.txt')
            roi = str(video / 'roi.jpg')


            sequence_output = os.path.join(folder_output, video.name)
            try:
                os.mkdir(sequence_output)

                out_frames = os.path.join(sequence_output, 'frames')

                print('Creating the frames.')
                save_frames(video_path, out_frames)
                print('Frames created.')

                shutil.copy(text_file, sequence_output)
                shutil.copy(calibration, sequence_output)
                shutil.copy(roi, sequence_output)
                
                # Execute yolov9
                os.system(f'python3 ../yolov9/detect.py --save-txt --save-conf --classes 0 1 2 3 5 7 --weights ../yolov9/weights/yolov9-c.pt --conf 0.5 --source {out_frames} --device 0')

                # Execute the file task_2.py
                os.system(f'python3 task2_tracking.py --det_path {yolo_runs} --path_imgs {out_frames} --path_output {sequence_output}')

                shutil.rmtree('../yolov9/runs/detect/exp/')
            except:
                pass






            





