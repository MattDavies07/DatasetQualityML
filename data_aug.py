import os
import numpy as np
from numpy.lib.function_base import percentile
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm # used as progress bar
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from skimage.segmentation import find_boundaries
from skimage.filters.rank import gradient
from skimage.morphology import disk, erosion, dilation
from clustering import getOutliers

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        files = glob(path + '*')
        for f in files:
            os.remove(f)

def load_data(path):
    train_x = sorted(glob(os.path.join(path, "training", "images", "*.tif")))
    train_y = sorted(glob(os.path.join(path, "training", "1st_manual", "*.gif")))

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.tif")))
    test_y = sorted(glob(os.path.join(path, "test", "1st_manual", "*.gif")))

    return (train_x, train_y), (test_x, test_y)

def augment_data(images, masks, save_path, augment=True, gradient=False, intensity=False):
    size = (512, 512)
    neighbourhood = disk(1)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        y = imageio.mimread(y)[0]

        if gradient == True:
            boundaries = find_boundaries(y)
            eroded = erosion(x, neighbourhood)
            dilated = dilation(x, neighbourhood)
            grad = dilated - eroded
            updated = grad * boundaries
            mask = updated == 0
            updated[mask] = y[mask]
            y = updated

        if intensity == True:
            boundaries = find_boundaries(y)
            y[boundaries == 1] = 255 - x[boundaries == 1]

        # Augmentation as small dataset
        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    data_path = "data/"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    """ Create directories to save the augmented data """
    create_dir("exp_data/train/image/")
    create_dir("exp_data/train/mask/")
    create_dir("exp_data/test/image/")
    create_dir("exp_data/test/mask/")

    """ Apply clustering threshold """
    # uncomment below section to apply clustering with percentile 
    """
    metric_data = pd.read_csv('data/training/pre_training_metrics.csv')
    percentile = 85
    outliers = getOutliers(metric_data, percentile)
    delImgArr = []
    for i, file in enumerate(train_x):
        filename = os.path.basename(file)
        if filename in outliers:
            delImgArr.append(file)
    for i in delImgArr:
        train_x.remove(i)
        filename = os.path.basename(i)
        mask_path = 'data/training/1st_manual'
        mask_file = filename[:2] + '_manual1.gif'
        mask = os.path.join(mask_path, mask_file)
        train_y.remove(mask)
    """
    """ Apply intensity measure """


    """ Data augmentation """
    augment_data(train_x, train_y, "exp_data/train/", augment=True, gradient=False, intensity=False)
    augment_data(test_x, test_y, "exp_data/test/", augment=False, gradient=False, intensity=False) # don't apply augmentation to testing

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")