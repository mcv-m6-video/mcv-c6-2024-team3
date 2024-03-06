import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

def create_gif_with_bounding_boxes(frames_folder, detections_file, output_gif):
    # Read detections file
    detections = []
    with open(detections_file, 'r') as file:
        for line in file:
            frame_id, track_id, left, top, width, height, _, _, _, _ = line.split(',')
            frame_id = int(float(frame_id[-5:]))  # Convert floating-point frame ID to int
            detections.append((frame_id+1, int(float(track_id)), int(float(left)), int(float(top)), int(float(width)), int(float(height))))


# Extract unique track IDs
    track_ids = list(set(detection[1] for detection in detections))

    # Generate a color map
    color_map = {}
    for track_id in track_ids:
        color_map[track_id] = np.random.rand(3,)

    # Load frames
    frame_files = sorted(os.listdir(frames_folder))

    # Initialize Matplotlib figure
    fig, ax = plt.subplots()

    # Iterate through frames
    images = []
    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(frame)

        # Filter detections for the current frame
        frame_detections = [detection for detection in detections if detection[0] == int(frame_file.split('.')[0][-5:])]

        # Draw bounding boxes
        for detection in frame_detections:
            frame_id, track_id, left, top, width, height = detection
            color = color_map[track_id]
            rect = Rectangle((left, top), width, height, linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(left, top, str(track_id), color='white', fontsize=8, verticalalignment='top')

        # Save the image to a buffer
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.margins(0, 0)
        plt.draw()
        plt.pause(0.001)  # Pause needed for drawing to complete
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)

        # Clear axes for next frame
        ax.clear()

    # Save images as a GIF
    imageio.mimsave(output_gif, images, fps=10)

    plt.close()



# Usage
frames_folder = "./data_yolo/images"
detections_file = "./tracking/tracking_DeepSort.txt"
output_gif = "output.gif"

create_gif_with_bounding_boxes(frames_folder, detections_file, output_gif)
