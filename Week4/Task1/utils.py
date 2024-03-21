import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2
from tqdm import tqdm
import math

def real_world_coord(homography, x_value, y_value):
    
    #Get homogeneus coordinates
    point_hg = np.array([x_value, y_value, 1])

    realworld_pixel = np.dot(homography, point_hg)

    # Convert back to pixel coordinates
    realworld_pixel_x = realworld_pixel[0] / realworld_pixel[2]
    realworld_pixel_y = realworld_pixel[1] / realworld_pixel[2]

    return realworld_pixel_x, realworld_pixel_y



def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the Earth's surface given their latitude and longitude
    in decimal degrees.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    radius_earth = 6371  # Radius of Earth in kilometers. Use 3956 for miles
    distance = radius_earth * c

    return distance


def read_ground_truth(xml_file, classes, n_frames):
    '''
    [
        [[x1, y1, x2, y2], [x1, y1, x2, y2], …],
        [[x1, y1, x2, y2], [x1, y1, x2, y2], …], 
        …,
        [[...]]
    ]
    '''
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bbox_info = [[] for _ in range(n_frames)]

    print(len(bbox_info))

    for track in root.findall('./track'):
        label = track.attrib['label']

        if label in classes:
            for box in track.findall('box'):
                parked = False
                for attribute in box.findall('attribute'):
                    if attribute.attrib.get('name') == 'parked' and attribute.text == 'true':
                        parked = True
                        break
                if not parked:
                    id = track.attrib['id']
                    frame = int(box.attrib['frame'])
                    xtl = float(box.attrib['xtl'])
                    ytl = float(box.attrib['ytl'])
                    xbr = float(box.attrib['xbr'])
                    ybr = float(box.attrib['ybr'])
                    
                    bbox_info[frame].append([
                        xtl, ytl, xbr, ybr, id
                    ])
        
    return bbox_info

def get_speed(path, homography, frame_list, frameNumber, bboxGT, last_coords):
    '''
    paht don vols treure la foto per imrpimir en color
    frame_list la llista amb els noms dels frames
    framenumber
    bboxGT la llista de llistes amb les coordenades dels bbox [[x1, y1, x2, y2], [x1, y1, x2, y2], …]
    bboxPred la llista de llistes amb les coordenades dels bbox [[x1, y1, x2, y2], [x1, y1, x2, y2], …]
    '''
    frame = cv2.imread(path + frame_list[frameNumber], cv2.IMREAD_COLOR)
    print(last_coords)
    
    for bbox in bboxGT:
        xtl, ytl, xbr, ybr, id = bbox
        xtl, ytl, xbr, ybr, id = int(xtl), int(ytl), int(xbr), int(ybr), int(id)

        if id == 13:
            
            if not frameNumber in last_coords:
                last_coords[frameNumber] = list()

            #Get box center
            x_center = int((xtl + xbr) / 2)
            y_center = int((ytl + ybr) / 2)
            
            #Compute its GPS position
            realworld_pixel_x, realworld_pixel_y = real_world_coord(homography, x_center, y_center)

            last_coords[frameNumber].append([realworld_pixel_x, realworld_pixel_y])
            
            # text = f'X: {realworld_pixel_x} | Y: {realworld_pixel_y}'

            speed = haversine_distance(realworld_pixel_x, realworld_pixel_y, last_coords[0], last_coords[1])

            text = f'Speed: {speed} km/h'

            # Calculate text position
            text_x = xtl
            text_y = ytl - 5  # Adjust this value to change the distance between text and bbox

            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)
    
    # for bbox in bboxPred:
    #     xtl, ytl, xbr, ybr = bbox
    #     xtl, ytl, xbr, ybr = int(xtl), int(ytl), int(xbr), int(ybr)
    #     cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 0, 255), 2)
    
    
    cv2.imwrite('resultsBBGT/' + frame_list[frameNumber], frame)

    return realworld_pixel_x, realworld_pixel_y