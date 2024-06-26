import numpy as np
from typing import List

'''
All the code is based on this repository: https://github.com/mcv-m6-video/mcv-m6-2023-team2/blob/main/week1/metrics.py
and the Team 4 2024: https://github.com/mcv-m6-video/mcv-c6-2024-team4/
'''

def voc_ap(rec, prec):
    """
    Compute VOC AP given precision and recall using the VOC 07 11-point method.

    Original code from https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/pascal_voc_evaluation.py
    """
    ap = 0.0
    
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.0

    return ap

def voc_iou(pred: List[int], gt: np.ndarray):
    """
    Calculate IoU between detect box and gt boxes.
    :param pred: Predicted bounding box coordinates [x1, y1, x2, y2].
    :param gt: Ground truth bounding box coordinates [[x1, y1, x2, y2]].
    """
    # compute overlaps
    # intersection
    ixmin = np.maximum(gt[:, 0], pred[0])
    iymin = np.maximum(gt[:, 1], pred[1])
    ixmax = np.minimum(gt[:, 2], pred[2])
    iymax = np.minimum(gt[:, 3], pred[3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inters = iw * ih

    # union
    uni = (
        (pred[2] - pred[0] + 1.0) * (pred[3] - pred[1] + 1.0)
        + (gt[:, 2] - gt[:, 0] + 1.0) * (gt[:, 3] - gt[:, 1] + 1.0)
        - inters
    )

    return inters / uni

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
        bbox = np.array([[bbox[0], bbox[1], bbox[2], bbox[3]]  for bbox in frame])
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

    return ap, rec, prec

def mAP(gt, preds, N=10):
    aps = []
    recs = []
    precs = []
    for _ in range(N):
        ap, rec, prec = voc_eval(preds, gt)
        aps.append(ap)
        recs.append(rec)
        precs.append(prec)

    return np.mean(aps), np.mean(recs), np.mean(precs)
        