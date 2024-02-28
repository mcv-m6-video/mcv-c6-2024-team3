import cv2
import numpy as np

from task1_2 import *
from evaluation import *
import matplotlib.pyplot as plt


def voc_eval(preds, gt, ovthresh=0.5):
    """
    rec, prec, ap = voc_eval(preds,
                            gt,
                            [ovthresh],
                            )
    Top level function that does the PASCAL VOC evaluation.
    gt: Ground truth bounding boxes grouped by frames.
    preds: Predicted bounding boxes grouped by frames.
    [ovthresh]: Overlap threshold (default = 0.5)

    Original code from https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/pascal_voc_evaluation.py
    """
    # extract gt objects for this class
    class_recs = {}
    npos = 0

    for i, frame in enumerate(gt):
        bbox = np.array([[bbox[0], bbox[1], bbox[2], bbox[3]] for bbox in frame])
        difficult = np.array([False for bbox in frame]).astype(bool)
        det = [False] * len(frame)
        npos = npos + sum(~difficult)
        class_recs[i] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    image_ids = []
    confidence = []
    BB = []

    for i, frame in enumerate(preds):
        image_ids += [i] * len(preds[i])
        confidence += list(np.random.rand(len(preds[i])))
        BB += [[bbox[0], bbox[1], bbox[2], bbox[3]] for bbox in preds[i]]

    confidence = np.array(confidence)
    BB = np.array(BB).reshape(-1, 4)

    if np.all(confidence != None):
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    iou = np.zeros(nd)

    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            overlaps = voc_iou(bb, BBGT)
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
            iou[d] = ovmax

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)
    iou = np.mean(iou)

    return rec, prec, ap


def plot_prec_recall_curve(
    prec,
    rec,
    title="Precision-Recall curve",
    xAxis="Recall",
    yAxis="Precision",
    save_name="model.png",
):
    # plotting the points
    plt.plot(rec, prec)
    # naming the x axis
    plt.xlabel(xAxis)
    # naming the y axis
    plt.ylabel(yAxis)
    # giving a title to my graph
    plt.title(title)
    # function to show the plot
    plt.savefig(save_name)

    plt.close()


def evaluate(bboxPred, bboxGT):
    map = mAP(bboxPred, bboxGT)
    return map


def get_large_connected_components(binary_mask, min_area_threshold):
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    # Filter out the large connected components based on area
    large_components = []
    bounding_boxes = []
    for label in range(1, num_labels):  # Skip background label 0
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area_threshold and area < 200_000:
            large_components.append(labels == label)
            # Get the bounding box coordinates
            x, y, w, h = cv2.boundingRect((labels == label).astype(np.uint8))
            bounding_boxes.append([x, y, x + w, y + h])

    return bounding_boxes


# Create VideoCapture object

roi = cv2.imread("roi.jpg", cv2.IMREAD_GRAYSCALE)
min_area_threshold = 1000  # Adjust as needed
xml_file = "ai_challenge_s03_c010-full_annotation.xml"
classes = ["car"]
gt_bboxes = read_ground_truth(xml_file, classes, 2141)


for model in ["MOG2", "KNN", "GMG", "GSOC", "LSBP", "CNT"]:
    print(f"### {model} ###")
    if model == "MOG2":
        background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    elif model == "KNN":
        background_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=False)
    elif model == "CNT":
        background_subtractor = cv2.bgsegm.createBackgroundSubtractorCNT()
    elif model == "GMG":
        background_subtractor = cv2.bgsegm.createBackgroundSubtractorGMG()
    elif model == "GSOC":
        background_subtractor = cv2.bgsegm.createBackgroundSubtractorGSOC()
    elif model == "LSBP":
        background_subtractor = cv2.bgsegm.createBackgroundSubtractorLSBP()
    else:
        raise ValueError(
            "Invalid method. Please choose from 'MOG2', 'KNN', 'CNT', 'GMG', 'GSOC', 'LSBP'."
        )
    cap = cv2.VideoCapture("/home/georg/projects/mcv-c6-2024-team3/Week1/vdo.avi")
    # Create VideoWriter object to save output video
    output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"{model}.mp4", fourcc, fps, (output_width, output_height))

    pred_bboxes = []
    frame_counter = 0
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Apply background subtraction
        fg_mask = background_subtractor.apply(frame)
        fg_mask = cv2.bitwise_and(fg_mask, roi)
        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )

        bounding_boxes = get_large_connected_components(fg_mask, min_area_threshold)
        pred_bboxes.append(bounding_boxes)
        # Overlay bounding boxes on the frame

        for bbox in bounding_boxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for bbox in gt_bboxes[frame_counter]:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame)
        frame_counter += 1

    # Release VideoCapture and VideoWriter objects
    cap.release()
    out.release()

    rec, prec, ap = voc_eval(preds=pred_bboxes, gt=gt_bboxes)
    print(rec, prec, ap)
    plot_prec_recall_curve(
        prec=prec, rec=rec, title=f"Precision-Recall {model}", save_name=f"{model}.png"
    )
