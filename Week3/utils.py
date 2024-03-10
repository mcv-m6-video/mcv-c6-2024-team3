import cv2
import shutil
import os


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