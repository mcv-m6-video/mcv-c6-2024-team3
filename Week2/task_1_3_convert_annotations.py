import xml.etree.ElementTree as ET
import os
import shutil
import cv2


def read_ground_truth(xml_file, classes, n_frames):
    """
    [
        [[x1, y1, x2, y2], [x1, y1, x2, y2], …],
        [[x1, y1, x2, y2], [x1, y1, x2, y2], …],
        …,
        [[...]]
    ]
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bbox_info = [[] for _ in range(n_frames)]

    print(len(bbox_info))

    for track in root.findall("./track"):
        label = track.attrib["label"]

        if label in classes.keys():
            for box in track.findall("box"):

                frame = int(box.attrib["frame"])
                xtl = float(box.attrib["xtl"])
                ytl = float(box.attrib["ytl"])
                xbr = float(box.attrib["xbr"])
                ybr = float(box.attrib["ybr"])
                label_int = classes[label]

                bbox_info[frame].append([label_int, xtl, ytl, xbr, ybr])

    return bbox_info


def xyxy2yolo(coords, img_h, img_w):
    cls, x1, y1, x2, y2 = coords
    x = ((x1 + x2) / 2) / img_w
    y = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    return [cls, x, y, w, h]


if __name__ == "__main__":

    annotation_file = "./ai_challenge_s03_c010-full_annotation.xml"
    video_file = "./vdo.avi"
    classes = {"car": 2, "bike": 1}

    write_dest = "./dataset"
    os.makedirs(write_dest)
    os.makedirs(write_dest + "/images")
    os.makedirs(write_dest + "/labels")

    gt_bboxes = read_ground_truth(annotation_file, classes=classes, n_frames=2140)

    cap = cv2.VideoCapture(video_file)
    # Loop through each frame
    frame_nr = 0
    while True:
        ret, frame = cap.read()
        im_h, im_w, _ = frame.shape
        frame_bboxes = gt_bboxes[frame_nr]
        yolo_boxes = [xyxy2yolo(bbox, im_h, im_w) for bbox in frame_bboxes]

        frame_file_string = "{:06d}".format(frame_nr)
        frame_dest = os.path.join(write_dest, "images", frame_file_string + ".png")
        cv2.imwrite(frame_dest, frame)

        yolo_boxes = [
            [str(box[0]), str(box[1]), str(box[2]), str(box[3]), str(box[4])]
            for box in yolo_boxes
        ]

        yolo_boxes = [" ".join(box) for box in yolo_boxes]

        write_string = "\n".join(yolo_boxes)

        annotation_dest = frame_dest.replace("images", "labels").replace("png", "txt")
        with open(annotation_dest, "w") as fp:
            fp.write(write_string)
        frame_nr += 1
