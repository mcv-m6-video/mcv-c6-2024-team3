from torchvision.io.image import read_image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchmetrics.detection import MeanAveragePrecision

from tqdm import tqdm

import os
import torch


def yolo2xyxy(coords, height, width):
    cls = int(coords[0])
    x, y, w, h = [float(coord) for coord in coords[1:]]
    x1 = (x - w / 2) * width
    y1 = (y - h / 2) * height
    x2 = (x + w / 2) * width
    y2 = (y + h / 2) * height

    return [cls, x1, y1, x2, y2]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images_dir = "/ghome/group04/georg-c6-w2/dataset/images"
image_paths = [
    os.path.join(images_dir, image_file) for image_file in os.listdir(images_dir)
]

# Step 1: Initialize model with the best available weights
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model = model.to(device)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
preds = []
gt_s = []
for image_path in tqdm(image_paths):

    label_path = image_path.replace("images", "labels").replace(".png", ".txt")

    img = read_image(image_path).to(device)
    _, h, w = img.shape

    with open(label_path, "r") as fp:
        lines = [line.split() for line in fp.readlines()]
        lines = [yolo2xyxy(line, h, w) for line in lines]

    boxes = []
    labels = []

    for line in lines:
        labels.append(line[0] + 1)
        boxes.append(line[1:])

    gt_dict = {
        "boxes": torch.tensor(boxes, device=device),
        "labels": torch.tensor(labels, device=device),
    }
    gt_s.append(gt_dict)

    batch = [preprocess(img)]

    # Step 4: Use the model and visualize the prediction
    with torch.no_grad():
        prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]

    preds.append(prediction)

    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    img = draw_bounding_boxes(
        img,
        boxes=prediction["boxes"],
        labels=labels,
        colors="red",
        width=4,
        font_size=30,
    )

    labels = [weights.meta["categories"][i] for i in gt_dict["labels"]]
    box = draw_bounding_boxes(
        img,
        boxes=gt_dict["boxes"],
        labels=labels,
        colors="blue",
        width=4,
        font_size=30,
    )
    img = to_pil_image(box.detach())
    img.save("faster-r-cnn/" + image_path.split("/")[-1], "JPEG")

metric = MeanAveragePrecision(iou_type="bbox")
metric.update(preds, gt_s)
from pprint import pprint

pprint(metric.compute())
