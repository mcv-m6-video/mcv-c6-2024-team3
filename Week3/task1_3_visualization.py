import cv2
import numpy as np
from PIL import Image
import pickle
from pathlib import Path

from moviepy.editor import VideoFileClip

def convert_to_gif(input_file, output_file, fps=5, scale=None):
    # Load the video clip
    clip = VideoFileClip(input_file)
    
    # Adjust the scale if provided
    if scale:
        clip = clip.resize(scale)
    
    # Write the GIF file with loop set to 0 for infinite looping
    clip.write_gif(output_file, fps=fps, loop=0)


output_video = 'tracking_yolo_sort/optical_flow.mp4'

im1 = np.array(Image.open('tracking_yolo_sort/frames/frame00000.png'))

cap = cv2.VideoCapture('../OptionalTaskW2Dataset/train/S03/c010/vdo.avi')
    
# Get the frames per second
fps = cap.get(cv2.CAP_PROP_FPS)

# Release the video capture object
cap.release()

pickle_files = Path('tracking_yolo_sort/optical_flow/')

height, width, _ = im1.shape
    
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID' on Windows
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for pickle_file in pickle_files.glob('*.pkl'):

    with open(pickle_file, 'rb') as file:
        flow = pickle.load(file)

    hsv = np.zeros(im1.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    out.write(rgb)

out.release()


convert_to_gif(output_video, output_video.replace('.mp4', '.gif'), fps=2, scale=0.25)
