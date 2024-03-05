import os
import cv2 


def yolo2xyxy(line, im_h, im_w):
    class_, x, y, w, h = line
    cls = int(class_)
    
    x1 = int((float(x) - float(w)/2) * im_w)
    x2 = int((float(x) + float(w)/2) * im_w)
    y1 = int((float(y) - float(h)/2) * im_h)
    y2 = int((float(y) + float(h)/2) * im_h)

    return [cls, x1, y1, x2, y2]

def draw_rectangles(img, lines, color):
    for line in lines:
        cv2.rectangle(img, (line[1], line[2]), (line[3], line[4]), color=color, thickness=2)


images_dir = '/ghome/group04/MCV-C5-G4/week2/optional_task/dataset/test/images'
gt_dir = '/ghome/group04/MCV-C5-G4/week2/optional_task/dataset/test/labels'
pred_dir = '/ghome/group04/MCV-C5-G4/week2/optional_task/yolov9/runs/val/yolov9_c_c_640_val7/labels'

visualization_dir = './visualizations'

#os.mkdir(visualization_dir)

images = sorted(os.listdir(images_dir))
gt_s = [image.replace('png', 'txt') for image in images]
preds = gt_s

for image, gt, pred in zip(images, gt_s, preds):
    img_path = os.path.join(images_dir, image)
    gt_path = os.path.join(gt_dir, gt)
    pred_path = os.path.join(pred_dir, pred)

    img = cv2.imread(img_path)
    im_h, im_w, _ = img.shape
    with open(gt_path, 'r') as fp:
        gt = [line.strip().split() for line in fp.readlines()]

    with open(pred_path, 'r') as fp:
        pred = [line.strip().split() for line in fp.readlines()]

    xyxy_gt = [yolo2xyxy(line, im_h, im_w) for line in gt]
    xyxy_pred = [yolo2xyxy(line, im_h, im_w) for line in pred]

    draw_rectangles(img=img, lines = xyxy_gt, color = (255, 0, 0))
    draw_rectangles(img=img, lines = xyxy_pred, color = (0, 0, 255))

    cv2.imwrite(os.path.join(visualization_dir, image), img)