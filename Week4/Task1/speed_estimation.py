import cv2
import numpy as np
from utils import *
import os

#Videos are at 10fps
xml_file = '/ghome/group02/C6/Week1/ai_challenge_s03_c010-full_annotation.xml'

classes = ['bike'] # The other class is bike

path = '/ghome/group02/C6/Week1/framesOriginal/'
output_folder = 'resultsBBGT/'
frames_list = sorted(os.listdir(path))

n_frames = len(frames_list)
bbox_info = read_ground_truth(xml_file, classes, n_frames)

print(len(bbox_info))

#Homography matrix
homography_matrix = np.array([[-82.3017308, 159.6429809, 17973.0298053],
                              [-4.1691749, 2.0775241, 365.5647553],
                              [0.0134773, 0.0173450, 1.0000000]])

#Get the inverse
homography_matrix = np.linalg.inv(homography_matrix)
print(homography_matrix)

last_coords = [0,0]
speed = 0
avg_frames = 5 #Every how many frames do you want to compute the speed

# for frameNumber in tqdm(range(len(frames_list))):
for frameNumber in tqdm(range(0,200)): #Process first 200 frames
    
    frame = cv2.imread(path + frames_list[frameNumber], cv2.IMREAD_COLOR)
    # print(last_coords)
    bboxes = bbox_info[frameNumber]
    
    for bbox in bboxes:
        xtl, ytl, xbr, ybr, id = bbox
        xtl, ytl, xbr, ybr, id = int(xtl), int(ytl), int(xbr), int(ybr), int(id)

        if id == 7: #Just 1 id, to make it simple

            #Get box center
            if frameNumber%avg_frames == 0:
                x_center = int((xtl + xbr) / 2)
                y_center = int((ytl + ybr) / 2)

                print('BBOX: ', xtl, ytl, xbr, ybr, x_center, y_center)
                
                #Compute its GPS position
                realworld_pixel_x, realworld_pixel_y = real_world_coord(homography_matrix, x_center, y_center)
                
                # text = f'X: {realworld_pixel_x} | Y: {realworld_pixel_y}'

                #Compute the distance between the coordinates on this frame respect the last one
                distance = haversine_distance(realworld_pixel_x, realworld_pixel_y, last_coords[0], last_coords[1])*1000
                print(distance)

                #Compute distance
                speed = (distance / (1/10))/avg_frames #Its 1/10 because the video has 10fps
                speed = round((speed*3600)/1000, 2)

                #Save these coordinates as the last ones
                last_coords = [realworld_pixel_x, realworld_pixel_y]

            text = f'Speed: {speed} km/h'

            # Calculate text position
            text_x = xtl
            text_y = ytl - 5  # Adjust this value to change the distance between text and bbox

            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)
    
    # for bbox in bboxPred:
    #     xtl, ytl, xbr, ybr = bbox
    #     xtl, ytl, xbr, ybr = int(xtl), int(ytl), int(xbr), int(ybr)
    #     cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 0, 255), 2)
    
    
    cv2.imwrite(output_folder + frames_list[frameNumber], frame)


