from ctypes import util
import cv2
import shutil
import os

from utils import reset_folder, save_frames

if __name__ == "__main__":
    input_video = 'AICity_data_S04_C016/train/S04/c016/vdo.avi'
    output_frames = 'video_frames'

    if os.path.exists(output_frames): # Estoy suponiendo que ya he generado las imagenes pa no tener que hacerlo otra vez
        save_frames(input_video, output_frames)