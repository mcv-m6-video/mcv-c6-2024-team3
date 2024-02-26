import xml.etree.ElementTree as ET
import os

import cv2

def read_ground_truth(xml_file, classes):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bbox_info = {}

    for track in root.findall('./track'):
        label = track.attrib['label']
        if label in classes:
            for box in track.findall('box'):
                frame = int(box.attrib['frame'])
                xtl = float(box.attrib['xtl'])
                ytl = float(box.attrib['ytl'])
                xbr = float(box.attrib['xbr'])
                ybr = float(box.attrib['ybr'])
                
                bbox_info.setdefault(frame, []).append([
                    xtl, ytl, xbr, ybr
                ])
        
    return bbox_info

def draw_bbox(path, frame_list, frameNumber, bboxGT, bboxPred, draw = True):
    frame = cv2.imread(path + frame_list[frameNumber], cv2.IMREAD_COLOR)
    
    for bbox in bboxGT:
        xtl, ytl, xbr, ybr = bbox
        xtl, ytl, xbr, ybr = int(xtl), int(ytl), int(xbr), int(ybr)
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)
    
    for bbox in bboxPred:
        xtl, ytl, xbr, ybr = bbox
        xtl, ytl, xbr, ybr = int(xtl), int(ytl), int(xbr), int(ybr)
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 0, 255), 2)
    
    if draw:
        cv2.imshow('frame', frame)
        cv2.waitKey(0)

if __name__ == "__main__":
    xml_file = 'ai_challenge_s03_c010-full_annotation.xml'

    classes = ['car'] # The other class is bike
    bbox_info = read_ground_truth(xml_file, classes)

    frameNumber = 1000
    path = 'framesOriginal/'
    frames_list = sorted(os.listdir(path))

    print(frames_list)
    draw_bbox(path, frames_list, frameNumber, bbox_info[frameNumber], [])

