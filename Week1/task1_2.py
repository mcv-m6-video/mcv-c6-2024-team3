import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2
from tqdm import tqdm
from evaluation import voc_ap, voc_iou, voc_eval

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
                    frame = int(box.attrib['frame'])
                    xtl = float(box.attrib['xtl'])
                    ytl = float(box.attrib['ytl'])
                    xbr = float(box.attrib['xbr'])
                    ybr = float(box.attrib['ybr'])
                    
                    bbox_info[frame].append([
                        xtl, ytl, xbr, ybr
                    ])
        
    return bbox_info

def draw_bbox(path, frame_list, frameNumber, bboxGT, bboxPred):
    '''
    paht don vols treure la foto per imrpimir en color
    frame_list la llista amb els noms dels frames
    framenumber
    bboxGT la llista de llistes amb les coordenades dels bbox [[x1, y1, x2, y2], [x1, y1, x2, y2], …]
    bboxPred la llista de llistes amb les coordenades dels bbox [[x1, y1, x2, y2], [x1, y1, x2, y2], …]
    '''
    frame = cv2.imread(path + frame_list[frameNumber], cv2.IMREAD_COLOR)
    
    for bbox in bboxGT:
        xtl, ytl, xbr, ybr = bbox
        xtl, ytl, xbr, ybr = int(xtl), int(ytl), int(xbr), int(ybr)
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)
    
    for bbox in bboxPred:
        xtl, ytl, xbr, ybr = bbox
        xtl, ytl, xbr, ybr = int(xtl), int(ytl), int(xbr), int(ybr)
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 0, 255), 2)
    
    
    cv2.imshow('frame', frame)
    cv2.waitKey(0)

def get_bbox_predictions(path, frame_list):
    '''
    FUNCTION FOR JUST SEE THE PLOTS
    '''
    for frameNumber in range(len(frame_list)):

        bbox_pred_info = []
        
        image = cv2.imread(path + frame_list[frameNumber], cv2.IMREAD_COLOR)

        if len(image.shape) > 2:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

        min_area_threshold = 100

        filtred_components = np.zeros_like(gray_image)

        for i in range(1, num_labels):  # Exclude background label which is 0
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area_threshold:
                mask = labels == i
                color = 255  # Generate a random color
                filtred_components[mask] = color

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filtred_components, connectivity=8)

        _, binary_image = cv2.threshold(filtred_components, 127, 255, cv2.THRESH_BINARY)

        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30)))

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

        output_image = np.zeros_like(image)

        bounding_boxes = []

        for i in range(1, num_labels):  # Exclude background label which is 0
            area = stats[i, cv2.CC_STAT_AREA]
            mask = labels == i
            color = np.random.randint(0, 255, size=3)  # Generate a random color
            output_image[mask] = color

            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

            bounding_boxes.append((x, y, w, h))

        merged_boxes = []

        for box in bounding_boxes:
            x, y, w, h = box
            merged = False
            for index, (merged_x, merged_y, merged_w, merged_h) in enumerate(merged_boxes):
                if x >= merged_x and y >= merged_y and x + w <= merged_x + merged_w and y + h <= merged_y + merged_h:
                    merged_boxes[index] = (x, y, w, h)
                    merged = True
                    break
                elif x <= merged_x and y <= merged_y and x + w >= merged_x + merged_w and y + h >= merged_y + merged_h:
                    merged_boxes[index] = (merged_x, merged_y, merged_w, merged_h)
                    merged = True
                    break
            if not merged:
                bbox_pred_info.append([
                    x, y, x + w, y + h
                ])

        frame = cv2.imread(path + frame_list[frameNumber], cv2.IMREAD_COLOR)

        for bbox in bbox_pred_info:
            xtl, ytl, xbr, ybr = bbox
            xtl, ytl, xbr, ybr = int(xtl), int(ytl), int(xbr), int(ybr)
            cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 0, 255), 2)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)


def get_bbox_from_single_image(image):
    '''
    THE INPUT IS AN IMAGE IN GRAYSCALE COLORSPACE (THE PREDICTION OF THE MODEL) AND IT WIL RETURN A LIST OF LISTS IN THE FORM OF [bbox1, bbox2, ...]
    '''

    final_bounding_boxes = []

    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    min_area_threshold = 200

    filtred_components = np.zeros_like(gray_image)

    for i in range(1, num_labels):  # Exclude background label which is 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area_threshold:
            mask = labels == i
            color = 255  # Generate a random color
            filtred_components[mask] = color

    _, binary_image = cv2.threshold(filtred_components, 127, 255, cv2.THRESH_BINARY)

    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30)))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    output_image = np.zeros_like(image)

    bounding_boxes = []

    for i in range(1, num_labels):  # Exclude background label which is 0
        area = stats[i, cv2.CC_STAT_AREA]
        mask = labels == i
        color = np.random.randint(0, 255, size=3)  # Generate a random color
        output_image[mask] = color

        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

        bounding_boxes.append((x, y, w, h))

    merged_boxes = []

    for box in bounding_boxes:
        x, y, w, h = box
        merged = False
        for index, (merged_x, merged_y, merged_w, merged_h) in enumerate(merged_boxes):
            if x >= merged_x and y >= merged_y and x + w <= merged_x + merged_w and y + h <= merged_y + merged_h:
                merged_boxes[index] = (x, y, w, h)
                merged = True
                break
            elif x <= merged_x and y <= merged_y and x + w >= merged_x + merged_w and y + h >= merged_y + merged_h:
                merged_boxes[index] = (merged_x, merged_y, merged_w, merged_h)
                merged = True
                break
        if not merged:
            final_bounding_boxes.append([
                x, y, x + w, y + h
            ])

    return final_bounding_boxes

def get_all_bb(path_frames):
    '''
    THE INPUT IS THE PATH TO THE FOLDER WITH THE FRAMES AND IT WILL RETURN A LIST OF LISTS IN THE FORM OF [bbox1, bbox2, ...]
    '''
    frames_list = sorted(os.listdir(path_frames))
    bbox_info = []

    for frameNumber in tqdm(range(0, len(frames_list))):
        image = cv2.imread(path_frames + frames_list[frameNumber], cv2.IMREAD_COLOR)
        bbox_info.append(get_bbox_from_single_image(image))

    return bbox_info
    
    
'''
TO draw on a frame the GT and the predictions use the draw_bbox function
'''
if __name__ == "__main__":
    xml_file = 'ai_challenge_s03_c010-full_annotation.xml'

    classes = ['car'] # The other class is bike

    path = 'framesOriginal/'
    frames_list = sorted(os.listdir(path))

    n_frames = len(frames_list)
    bbox_info = read_ground_truth(xml_file, classes, n_frames)

    path = 'framesOriginal/'
    frames_list = sorted(os.listdir(path))

    print(len(bbox_info))

    frameNumber = 1000
    image = cv2.imread('framesResult/' + frames_list[frameNumber], cv2.IMREAD_COLOR)
    draw_bbox(path, frames_list, frameNumber, bbox_info[frameNumber], get_bbox_from_single_image(image))
    

    '''
    for frameNumber in range(len(frames_list)):
        image = cv2.imread('framesResult/' + frames_list[frameNumber], cv2.IMREAD_COLOR)
        draw_bbox(path, frames_list, frameNumber, bbox_info[frameNumber], get_bbox_from_single_image(image))
    '''

    '''
    path_results = 'framesResult_adaptive/'
    frames_list_results = sorted(os.listdir(path_results))
    bbox_info_predictions = get_bbox_predictions(path_results, frames_list_results)
    '''
    


