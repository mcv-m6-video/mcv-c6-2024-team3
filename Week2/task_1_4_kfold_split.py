import os
import shutil
import random

dataset_dir = "dataset"

dest_dir = "dataset/splits_B/"
# dest_dir = "dataset/splits_C/"
strategy = "B"


def make_images_labels_dirs(directory):
    os.makedirs(os.path.join(directory, "images"))
    os.makedirs(os.path.join(directory, "labels"))


imgs_src_dir = os.path.join(dataset_dir, "images")
labs_src_dir = os.path.join(dataset_dir, "labels")

image_files = os.listdir(imgs_src_dir)
if strategy == "B":
    image_files = sorted(
        image_files
    )  # sort files by name for k-fold strategy B in slides
elif strategy == "C":
    random.shuffle(image_files)  # shuffle files for k-fold strategy C in slides

label_files = [file.replace("png", "txt") for file in image_files]


num_folds = 4
len_fold = len(image_files) // num_folds
for k in range(0, num_folds):

    fold_dir = os.path.join(dest_dir, str(k))

    fold_train_dir = os.path.join(fold_dir, "train")
    make_images_labels_dirs(fold_train_dir)

    fold_test_dir = os.path.join(fold_dir, "test")
    make_images_labels_dirs(fold_test_dir)

    fold_start_index = len_fold * k
    fold_end_index = len_fold * (k + 1)

    for i, image in enumerate(image_files):

        if i >= fold_start_index and i < fold_end_index:
            img_src = os.path.join(imgs_src_dir, image)
            lab_src = os.path.join(labs_src_dir, label_files[i])

            img_dest = os.path.join(fold_train_dir, "images", image)
            lab_dest = os.path.join(fold_train_dir, "labels", label_files[i])

            shutil.copy(img_src, img_dest)
            shutil.copy(lab_src, lab_dest)

        else:
            img_src = os.path.join(imgs_src_dir, image)
            lab_src = os.path.join(labs_src_dir, label_files[i])

            img_dest = os.path.join(fold_test_dir, "images", image)
            lab_dest = os.path.join(fold_test_dir, "labels", label_files[i])

            shutil.copy(img_src, img_dest)
            shutil.copy(lab_src, lab_dest)
