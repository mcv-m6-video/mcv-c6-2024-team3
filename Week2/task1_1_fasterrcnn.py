import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import os
from utils import reset_folder

if __name__ == '__main__':

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights = weights, box_score_thresh = 0.1)
    model.eval()

    input_path = 'framesOriginal'
    image_list = os.listdir(input_path)

    output_files = 'maskRCNN_output'

    reset_folder(output_files)

    for img_name in image_list:
        with open(os.path.join(output_files, img_name.split('.')[0] + '.txt'), 'w') as f:
            img_path = os.path.join(input_path, img_name)

            img = read_image(img_path)

            c, h, w = img.shape

            preprocess = weights.transforms()

            batch = [preprocess(img)]

            predictions = model(batch)[0]

            labels = [weights.meta['categories'][i] for i in predictions['labels']]

            for bbox, label, score, class_id in zip(predictions['boxes'], labels, predictions['scores'].detach(), predictions['labels'].detach()):

                xmin, ymin, xmax, ymax = bbox.detach()[0], bbox.detach()[1], bbox.detach()[2], bbox.detach()[3]
                class_name = label
                conf = score.item()
                class_id = class_id.item()

                x_centre, y_centre = (xmin + xmax) / 2, (ymin + ymax) / 2
                width, height = xmax - xmin, ymax - ymin

                # We need to normalize the coordinates
                x_centre /= w
                y_centre /= h
                width /= w
                height /= h

                if class_name in ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person']:
                    f.write(f"{class_id - 1} {x_centre} {y_centre} {width} {height} {conf}\n")

